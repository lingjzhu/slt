from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from .data import DEFAULT_DATASET_CONFIGS, DatasetConfig
from .gesture_data import GestureSignClipCollator, build_dataloader, build_eval_dataset, load_label_bank
from .gesture_model import GestureSignCLIPModel
from .metrics import AccuracyCounts


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--datasets", nargs="+", default=None, help="Subset of dataset names to evaluate.")
    return p.parse_args()


@torch.no_grad()
def encode_texts(core, texts, *, batch_size, max_text_length, device):
    tokenizer = core.tokenizer
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        feats = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        feats = {k: v.to(device) for k, v in feats.items()}
        out.append(core.encode_text(feats).float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate_split(core, config: DatasetConfig, train_args: dict, *, device, eval_batch_size: int, num_workers: int):
    test_config = DatasetConfig(
        name=config.name,
        root=config.root,
        language=config.language,
        train_split=config.train_split,
        eval_split="test",
        keep_native_fps=config.keep_native_fps,
    )
    collator = GestureSignClipCollator(core.tokenizer, max_text_length=train_args["max_text_length"])
    dataset = build_eval_dataset(
        test_config,
        manifest_dir=train_args["manifest_dir"],
        data_root=train_args["data_root"],
        metadata_root=train_args["metadata_root"],
        max_frames=train_args["max_frames"],
        csl_val_ratio=train_args.get("csl_val_ratio", 0.02),
    )
    loader, _ = build_dataloader(
        dataset,
        collate_fn=collator,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    gesture_embs: list[torch.Tensor] = []
    sample_texts: list[str] = []
    for batch in loader:
        if batch is None:
            continue
        batch_tensors = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        emb = core.encode_gesture(
            batch_tensors["gesture"],
            gesture_attention_mask=batch_tensors["gesture_attention_mask"],
        )
        gesture_embs.append(emb.float().cpu())
        sample_texts.extend(batch["target_texts"])

    if not gesture_embs:
        return {"top1": 0.0, "top5": 0.0, "count": 0.0, "vocab_size": 0}

    gesture_matrix = torch.cat(gesture_embs, dim=0)
    label_bank = load_label_bank(test_config)
    vocab = [text for text in label_bank if text]
    text_to_idx = {t: i for i, t in enumerate(vocab)}
    text_bank = encode_texts(
        core,
        vocab,
        batch_size=eval_batch_size,
        max_text_length=train_args["max_text_length"],
        device=device,
    )
    valid_positions = [i for i, text in enumerate(sample_texts) if text in text_to_idx]
    if not valid_positions:
        return {"top1": 0.0, "top5": 0.0, "count": 0.0, "vocab_size": len(vocab)}

    gesture_matrix = gesture_matrix[valid_positions]
    targets = torch.tensor([text_to_idx[sample_texts[i]] for i in valid_positions], dtype=torch.long)
    sims = gesture_matrix @ text_bank.T

    counts = AccuracyCounts()
    counts.update(sims, targets)
    res = counts.as_dict()
    res["vocab_size"] = len(vocab)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    train_args = payload["args"]
    logger.info("loaded checkpoint step=%d", payload.get("step", -1))

    model = GestureSignCLIPModel(
        text_model_name=train_args.get("modernbert", "answerdotai/ModernBERT-base"),
        max_text_length=train_args["max_text_length"],
        feature_dim=train_args.get("feature_dim", 1104),
        max_frames=train_args["max_frames"],
        gesture_embed_dim=train_args.get("gesture_embed_dim", 512),
        gesture_depth=train_args.get("gesture_depth", 6),
        gesture_num_heads=train_args.get("gesture_num_heads", 8),
        gesture_mlp_ratio=train_args.get("gesture_mlp_ratio", 4.0),
        embedding_dim=train_args.get("embedding_dim", 512),
        projection_dropout=train_args.get("projection_dropout", 0.1),
        gradient_checkpointing=False,
        loss_type=train_args.get("loss_type", "infonce"),
    )
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()

    dataset_configs = list(DEFAULT_DATASET_CONFIGS)
    if args.datasets:
        dataset_configs = [c for c in dataset_configs if c.name in args.datasets]

    results: dict[str, dict[str, float]] = {}
    agg = AccuracyCounts()
    for config in dataset_configs:
        logger.info("evaluating %s/test", config.name)
        res = evaluate_split(
            model,
            config,
            train_args,
            device=device,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        logger.info(
            "%s -> top1=%.4f top5=%.4f count=%d vocab=%d",
            config.name,
            res["top1"],
            res["top5"],
            int(res["count"]),
            int(res["vocab_size"]),
        )
        results[config.name] = res
        agg.total += int(res["count"])
        agg.top1_correct += int(round(res["top1"] * res["count"]))
        agg.top5_correct += int(round(res["top5"] * res["count"]))

    results["overall"] = agg.as_dict()
    logger.info("overall -> %s", json.dumps(results["overall"]))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
