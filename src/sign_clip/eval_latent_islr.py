from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from .latent_data import (
    DEFAULT_LATENT_DATASET_CONFIGS,
    LatentSignClipCollator,
    build_dataloader,
    build_eval_dataset,
)
from .latent_model import LatentSignCLIPModel
from .metrics import AccuracyCounts


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=None)
    return parser.parse_args()


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
        feats = {key: value.to(device) for key, value in feats.items()}
        out.append(core.encode_text(feats).float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate_split(core, config, train_args: dict, *, device, eval_batch_size: int, num_workers: int):
    test_config = type(config)(
        name=config.name,
        root=config.root,
        language=config.language,
        train_split=config.train_split,
        eval_split="test",
        keep_native_fps=config.keep_native_fps,
    )
    collator = LatentSignClipCollator(core.tokenizer, max_text_length=train_args["max_text_length"])
    loader = build_dataloader(
        build_eval_dataset(test_config, num_frames=train_args["num_frames"]),
        collate_fn=collator,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    video_embs: list[torch.Tensor] = []
    sample_texts: list[str] = []
    for batch in loader:
        if batch is None:
            continue
        batch_tensors = {
            key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        emb = core.encode_video(
            batch_tensors["latents"],
            num_padding_frames=batch_tensors.get("num_padding_frames"),
        )
        video_embs.append(emb.float().cpu())
        sample_texts.extend(batch["target_texts"])

    if not video_embs:
        return {"top1": 0.0, "top5": 0.0, "count": 0.0, "vocab_size": 0}

    video_matrix = torch.cat(video_embs, dim=0)
    vocab = sorted({text for text in sample_texts if text})
    text_to_idx = {text: i for i, text in enumerate(vocab)}
    text_bank = encode_texts(
        core,
        vocab,
        batch_size=eval_batch_size,
        max_text_length=train_args["max_text_length"],
        device=device,
    )
    targets = torch.tensor([text_to_idx[text] for text in sample_texts], dtype=torch.long)
    sims = video_matrix @ text_bank.T

    counts = AccuracyCounts()
    counts.update(sims, targets)
    result = counts.as_dict()
    result["vocab_size"] = len(vocab)
    return result


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

    model = LatentSignCLIPModel(
        text_model_name=train_args.get("modernbert", "answerdotai/ModernBERT-base"),
        max_text_length=train_args["max_text_length"],
        embedding_dim=train_args["embedding_dim"],
        projection_dropout=train_args.get("projection_dropout", 0.1),
        gradient_checkpointing=False,
        num_frames=train_args["num_frames"],
        latent_channels=train_args.get("latent_channels", 16),
        latent_size=(train_args.get("latent_height", 28), train_args.get("latent_width", 28)),
        tubelet_size=(
            train_args.get("tubelet_frames", 2),
            train_args.get("tubelet_height", 4),
            train_args.get("tubelet_width", 4),
        ),
        vision_embed_dim=train_args.get("vision_embed_dim", 512),
        vision_depth=train_args.get("vision_depth", 8),
        vision_num_heads=train_args.get("vision_num_heads", 8),
        vision_mlp_ratio=train_args.get("vision_mlp_ratio", 4.0),
        drop_rate=train_args.get("drop_rate", 0.0),
        attn_drop_rate=train_args.get("attn_drop_rate", 0.0),
        drop_path_rate=train_args.get("drop_path_rate", 0.1),
        loss_type=train_args.get("loss_type", "infonce"),
    )
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()

    dataset_configs = list(DEFAULT_LATENT_DATASET_CONFIGS)
    if args.datasets:
        dataset_configs = [config for config in dataset_configs if config.name in args.datasets]

    results: dict[str, dict[str, float]] = {}
    aggregate = AccuracyCounts()
    for config in dataset_configs:
        logger.info("evaluating %s/test", config.name)
        result = evaluate_split(
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
            result["top1"],
            result["top5"],
            int(result["count"]),
            int(result["vocab_size"]),
        )
        results[config.name] = result
        aggregate.total += int(result["count"])
        aggregate.top1_correct += int(round(result["top1"] * result["count"]))
        aggregate.top5_correct += int(round(result["top5"] * result["count"]))

    results["overall"] = aggregate.as_dict()
    logger.info("overall -> %s", json.dumps(results["overall"]))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
