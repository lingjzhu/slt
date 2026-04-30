"""Training and evaluation for Qwen3 sign-language translation."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from .data import (
    Qwen3SLTCollator,
    _GESTURE_DATASET_LANGUAGE,
    build_dataloader,
    build_eval_dataset,
    build_train_dataset,
    discover_shards,
)
from .model import Qwen3SLT
from t5_slt.data import get_output_language_name, get_sign_language_name, normalize_text
from t5_slt.metrics import compute_translation_metrics, save_json

logger = logging.getLogger(__name__)


# ── DDP / utilities ──────────────────────────────────────────────────────────

def setup_ddp() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    from datetime import timedelta
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=60),
    )
    return True, rank, world, local_rank


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def parameter_count(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


def log_parameter_counts(model: torch.nn.Module) -> None:
    core = unwrap(model)
    total = parameter_count(core)
    trainable = parameter_count(core, trainable_only=True)
    logger.info(
        "qwen3_slt parameters: total=%s trainable=%s frozen=%s trainable_pct=%.2f%%",
        f"{total:,}", f"{trainable:,}", f"{total - trainable:,}",
        100.0 * trainable / max(1, total),
    )
    for name, module in core.named_children():
        child_total = parameter_count(module)
        if child_total:
            logger.info(
                "  %s: total=%s trainable=%s",
                name, f"{child_total:,}",
                f"{parameter_count(module, trainable_only=True):,}",
            )


def cosine_lr(step: int, *, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("step-*.pt"))
    return candidates[-1] if candidates else None


# ── Eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_split(
    *,
    split_name: str,
    model: torch.nn.Module,
    data_root: str | Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_frames: int,
    max_new_tokens: int,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    length_penalty: float,
    early_stopping: bool,
    max_eval_samples: int | None = None,
    dataset_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    core = unwrap(model)
    core.eval()
    tokenizer = core.tokenizer

    try:
        shards = discover_shards(data_root, split=split_name, dataset_names=list(dataset_names) if dataset_names else None)
    except FileNotFoundError as exc:
        logger.warning("[eval] no shards for %s: %s", split_name, exc)
        return {"split": split_name, "num_samples": 0}

    dataset = build_eval_dataset(shards, max_frames=max_frames)
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=min(2, num_workers),
        drop_last=False,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, Any]] = []
    n_seen = 0
    progress = tqdm(loader, desc=f"eval {split_name}", unit="batch", dynamic_ncols=True)

    for batch in progress:
        if batch is None:
            continue
        gesture = batch["gesture"].to(device, non_blocking=True)
        attn = batch["gesture_attention_mask"].to(device, non_blocking=True)

        gen = core.generate(
            gesture=gesture,
            gesture_attention_mask=attn,
            sign_languages=batch["sign_languages"],
            output_languages=batch["output_languages"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=early_stopping if num_beams > 1 else False,
        )
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for sample_id, url, ds_name, lang, sl, ol, pred, ref in zip(
            batch["sample_ids"],
            batch["urls"],
            batch["dataset_names"],
            batch["languages"],
            batch["sign_languages"],
            batch["output_languages"],
            decoded,
            batch["captions"],
        ):
            predictions.append({
                "sample_id": sample_id,
                "video_path": url,
                "dataset": ds_name,
                "language": lang,
                "prompt": f"Translate {sl} to {ol}",
                "prediction": normalize_text(pred, lang),
                "reference": normalize_text(ref, lang),
            })

        n_seen += gesture.size(0)
        progress.set_postfix(samples=n_seen)
        if max_eval_samples and n_seen >= max_eval_samples:
            break

    pred_path = output_dir / f"{split_name}_predictions.tsv"
    with pred_path.open("w", encoding="utf-8") as handle:
        handle.write("sample_id\tlanguage\tdataset\tprompt\tprediction\treference\tvideo_path\n")
        for row in predictions:
            handle.write(
                "\t".join([
                    row["sample_id"],
                    row["language"],
                    row["dataset"],
                    row["prompt"].replace("\t", " "),
                    row["prediction"].replace("\t", " "),
                    row["reference"].replace("\t", " "),
                    row["video_path"].replace("\t", " "),
                ]) + "\n"
            )

    result: dict[str, Any] = {"split": split_name, "num_samples": len(predictions)}
    if predictions:
        overall = compute_translation_metrics(
            [r["prediction"] for r in predictions],
            [r["reference"] for r in predictions],
        )
        by_language: dict[str, dict[str, float]] = {}
        by_dataset: dict[str, dict[str, float]] = {}
        grouped_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in predictions:
            grouped_lang[r["language"]].append(r)
            grouped_ds[r["dataset"]].append(r)
        for k, rows in grouped_lang.items():
            by_language[k] = compute_translation_metrics(
                [r["prediction"] for r in rows],
                [r["reference"] for r in rows],
            )
            by_language[k]["num_samples"] = len(rows)
        for k, rows in grouped_ds.items():
            by_dataset[k] = compute_translation_metrics(
                [r["prediction"] for r in rows],
                [r["reference"] for r in rows],
            )
            by_dataset[k]["num_samples"] = len(rows)
        result["overall"] = overall
        result["by_language"] = by_language
        result["by_dataset"] = by_dataset
        save_json(result, output_dir / f"{split_name}_metrics.json")
        logger.info("[eval %s] %s", split_name, json.dumps(overall))
    core.train()
    return result


# ── Train ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen3 SLT (gesture encoder + Qwen3 decoder).")
    # Data
    p.add_argument("--data-root", type=Path, default=Path("/mnt/data2/sign_gestures"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-names", nargs="+", default=None)
    p.add_argument("--max-frames", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--shuffle-buffer", type=int, default=2000)
    p.add_argument("--bucket-multiplier", type=int, default=4,
                   help="length-bucket buffer = bucket_multiplier * batch_size; 0 disables bucketing")
    # Model
    p.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--gesture-checkpoint", type=Path,
                   default=Path("/mnt/data4/outputs/gesture_pretraining_contrastive_150m/pretrained_encoder"))
    p.add_argument("--gesture-in-dim", type=int, default=1104)
    p.add_argument("--gesture-hidden-size", type=int, default=768)
    p.add_argument("--gesture-num-hidden-layers", type=int, default=22)
    p.add_argument("--gesture-num-attention-heads", type=int, default=12)
    p.add_argument("--gesture-intermediate-size", type=int, default=1152)
    p.add_argument("--gesture-max-position-embeddings", type=int, default=1024)
    p.add_argument("--gesture-global-attn-every-n-layers", type=int, default=3)
    p.add_argument("--gesture-local-attention", type=int, default=128)
    p.add_argument("--gesture-attn-impl", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--attn-impl", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--projection-hidden-dim", type=int, default=0,
                   help="0 → use Qwen3 hidden_size")
    p.add_argument("--projection-dropout", type=float, default=0.1)
    # Freeze controls
    p.add_argument("--unfreeze-encoder", action="store_true",
                   help="Unfreeze the gesture encoder.")
    p.add_argument("--unfreeze-projector", action="store_true", default=True,
                   help="Unfreeze the projector MLP (default: on).")
    p.add_argument("--freeze-projector", dest="unfreeze_projector", action="store_false")
    p.add_argument("--unfreeze-decoder", action="store_true",
                   help="Unfreeze the Qwen3 text decoder.")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-liger", dest="apply_liger", action="store_false", default=True)
    # Optimization
    p.add_argument("--batch-size", type=int, default=8, help="per-device train batch size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # Eval
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-max-samples", type=int, default=2000)
    p.add_argument("--eval-num-beams", type=int, default=4)
    p.add_argument("--eval-max-new-tokens", type=int, default=128)
    p.add_argument("--eval-repetition-penalty", type=float, default=1.0)
    p.add_argument("--eval-no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--eval-length-penalty", type=float, default=1.0)
    p.add_argument("--eval-early-stopping", action="store_true")
    p.add_argument("--eval-datasets", nargs="+", default=["how2sign_24fps", "csl_daily_24fps"])
    p.add_argument("--run-final-eval", action="store_true")
    # Checkpointing / logging
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="qwen3-slt")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    output_dir: Path, wandb_run_id: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"step-{step:07d}.pt"
    torch.save({
        "step": step,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "wandb_run_id": wandb_run_id,
    }, path)
    return path


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ddp_enabled, rank, world, local_rank = setup_ddp()
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        save_json({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  output_dir / "run_config.json")

    dataset_names = list(args.dataset_names) if args.dataset_names else None
    train_shards = discover_shards(args.data_root, split="train", dataset_names=dataset_names)
    if is_main:
        logger.info("Train shards: %d", len(train_shards))

    model = Qwen3SLT(
        model_name_or_path=args.model_name_or_path,
        gesture_in_dim=args.gesture_in_dim,
        gesture_hidden_size=args.gesture_hidden_size,
        gesture_num_hidden_layers=args.gesture_num_hidden_layers,
        gesture_num_attention_heads=args.gesture_num_attention_heads,
        gesture_intermediate_size=args.gesture_intermediate_size,
        gesture_max_position_embeddings=args.gesture_max_position_embeddings,
        gesture_global_attn_every_n_layers=args.gesture_global_attn_every_n_layers,
        gesture_local_attention=args.gesture_local_attention,
        gesture_attn_implementation=args.gesture_attn_impl,
        projection_hidden_dim=args.projection_hidden_dim or None,
        projection_dropout=args.projection_dropout,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        gesture_checkpoint=args.gesture_checkpoint,
        unfreeze_encoder=args.unfreeze_encoder,
        unfreeze_projector=args.unfreeze_projector,
        unfreeze_decoder=args.unfreeze_decoder,
        apply_liger=args.apply_liger,
    ).to(device)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()

    if is_main:
        log_parameter_counts(model)

    # Datasets
    bucket_size = args.batch_size * args.bucket_multiplier if args.bucket_multiplier > 0 else 0
    train_dataset = build_train_dataset(
        train_shards,
        max_frames=args.max_frames,
        shuffle_buffer=args.shuffle_buffer,
        length_bucket_size=bucket_size,
        batch_size=args.batch_size,
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # Optimizer (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters — set at least one of --unfreeze-{encoder,projector,decoder}")
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
    )
    optimizer.zero_grad(set_to_none=True)

    # Resume
    start_step = 0
    resume_path = Path(args.resume_from) if args.resume_from is not None else (
        find_latest_checkpoint(output_dir) if args.resume else None
    )
    wandb_run_id_prev = None
    if resume_path is not None and resume_path.exists():
        payload = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(payload["model"])
        try:
            optimizer.load_state_dict(payload["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        except ValueError as exc:
            if is_main:
                logger.warning("Skipped optimizer state from %s (%s)", resume_path, exc)
        start_step = int(payload.get("step", 0))
        wandb_run_id_prev = payload.get("wandb_run_id")
        if is_main:
            logger.info("Resumed from %s at step=%d", resume_path, start_step)
    if ddp_enabled:
        torch.distributed.barrier()

    # W&B
    wandb_run = None
    wandb_run_id = None
    if is_main and args.wandb:
        try:
            import wandb as _wandb
            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                dir=str(output_dir),
                id=wandb_run_id_prev,
                resume="allow" if wandb_run_id_prev else None,
            )
            wandb_run_id = wandb_run.id
        except Exception as exc:
            logger.warning("wandb init failed: %s", exc)

    accum_steps = max(1, args.gradient_accumulation_steps)
    train_iter = iter(train_loader)
    running = {"loss": 0.0}
    t0 = time.time()
    eval_dataset_names = tuple(args.eval_datasets) if args.eval_datasets else None

    for step in range(start_step + 1, args.train_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step - 1, base_lr=args.lr,
                                    warmup_steps=args.warmup_steps,
                                    total_steps=args.train_steps)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if batch is None:
            continue
        # Drop incomplete batches: WebDataset + DDP all-reduce requires uniform batch size.
        if batch["gesture"].shape[0] < args.batch_size:
            continue

        gesture = batch["gesture"].to(device, non_blocking=True)
        gattn = batch["gesture_attention_mask"].to(device, non_blocking=True)

        is_sync = step % accum_steps == 0
        sync_ctx = model.no_sync() if (ddp_enabled and not is_sync) else contextlib.nullcontext()
        with sync_ctx:
            outputs = model(
                gesture=gesture,
                gesture_attention_mask=gattn,
                sign_languages=batch["sign_languages"],
                output_languages=batch["output_languages"],
                target_texts=batch["captions"],
                max_target_length=args.max_target_length,
            )
            loss = outputs["loss"] / accum_steps
            loss.backward()

        running["loss"] += float(outputs["loss"].detach())

        if is_sync:
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if is_main and step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = running["loss"] / args.log_every
            dt = time.time() - t0
            ips = args.log_every * args.batch_size * world / max(dt, 1e-6)
            logger.info("step=%d loss=%.4f lr=%.2e samples/s=%.1f", step, avg_loss, lr, ips)
            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss": avg_loss, "train/lr": lr, "train/samples_per_s": ips},
                    step=step,
                )
            running = {k: 0.0 for k in running}
            t0 = time.time()

        if args.eval_every > 0 and step % args.eval_every == 0:
            if ddp_enabled:
                torch.distributed.barrier()
            if is_main:
                metrics = evaluate_split(
                    split_name="val",
                    model=model,
                    data_root=args.data_root,
                    output_dir=output_dir / "eval_predictions" / f"step-{step:07d}",
                    device=device,
                    batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    max_frames=args.max_frames,
                    max_new_tokens=args.eval_max_new_tokens,
                    num_beams=args.eval_num_beams,
                    repetition_penalty=args.eval_repetition_penalty,
                    no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
                    length_penalty=args.eval_length_penalty,
                    early_stopping=args.eval_early_stopping,
                    max_eval_samples=args.eval_max_samples,
                    dataset_names=eval_dataset_names,
                )
                if wandb_run is not None and metrics.get("overall"):
                    wandb_run.log({f"eval/{k}": v for k, v in metrics["overall"].items()}, step=step)
            if ddp_enabled:
                torch.distributed.barrier()

        if args.save_every > 0 and step % args.save_every == 0:
            if is_main:
                ckpt = save_checkpoint(model, optimizer, step, output_dir, wandb_run_id)
                logger.info("Saved checkpoint %s", ckpt)
            if ddp_enabled:
                torch.distributed.barrier()

    if is_main:
        ckpt = save_checkpoint(model, optimizer, args.train_steps, output_dir, wandb_run_id)
        logger.info("Saved final checkpoint %s", ckpt)
        unwrap(model).tokenizer.save_pretrained(output_dir / "tokenizer")

    if ddp_enabled:
        torch.distributed.barrier()

    if is_main and args.run_final_eval:
        for split in ("val", "test"):
            evaluate_split(
                split_name=split,
                model=model,
                data_root=args.data_root,
                output_dir=output_dir / "final_eval",
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                max_frames=args.max_frames,
                max_new_tokens=args.eval_max_new_tokens,
                num_beams=args.eval_num_beams,
                repetition_penalty=args.eval_repetition_penalty,
                no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
                length_penalty=args.eval_length_penalty,
                early_stopping=args.eval_early_stopping,
                max_eval_samples=None,
                dataset_names=eval_dataset_names,
            )

    if wandb_run is not None:
        wandb_run.finish()
    if ddp_enabled:
        torch.distributed.destroy_process_group()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
