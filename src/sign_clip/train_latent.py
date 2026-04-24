from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*_object_to_tensor.*")
warnings.filterwarnings("ignore", message=".*_tensor_to_object.*")
warnings.filterwarnings("ignore", module="webdataset.*")
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .latent_data import (
    DEFAULT_LATENT_DATASET_CONFIGS,
    LatentSignClipCollator,
    build_dataloader,
    build_eval_dataset,
    build_train_dataset,
    load_label_bank,
)
from .latent_model import LatentSignCLIPModel
from .metrics import AccuracyCounts


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modernbert", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sign_clip_latent"))
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-height", type=int, default=28)
    parser.add_argument("--latent-width", type=int, default=28)
    parser.add_argument("--tubelet-frames", type=int, default=2)
    parser.add_argument("--tubelet-height", type=int, default=4)
    parser.add_argument("--tubelet-width", type=int, default=4)
    parser.add_argument("--vision-embed-dim", type=int, default=512)
    parser.add_argument("--vision-depth", type=int, default=8)
    parser.add_argument("--vision-num-heads", type=int, default=8)
    parser.add_argument("--vision-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--max-text-length", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--sample-shuffle", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--equal-dataset-mix", action="store_true")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--wandb-project", default="sign-clip-latent-islr")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--loss-type", choices=("infonce", "sigmoid"), default="infonce")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def _setup_ddp() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
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
    return False, 0, 1, 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _make_autocast(device: torch.device, precision: str):
    if precision == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _split_text_features(batch: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("text_"): value
        for key, value in batch.items()
        if key.startswith("text_") and torch.is_tensor(value)
    }


def _cosine_lr(step: int, *, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _build_text_bank(
    model: torch.nn.Module,
    texts: list[str],
    *,
    batch_size: int,
    max_text_length: int,
    device: torch.device,
) -> tuple[list[str], dict[str, int], torch.Tensor]:
    core = _unwrap(model)
    tokenizer = core.tokenizer
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        features = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        features = {key: value.to(device) for key, value in features.items()}
        embeddings.append(core.encode_text(features).float().cpu())
    bank = torch.cat(embeddings, dim=0)
    index = {text: i for i, text in enumerate(texts)}
    return texts, index, bank


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    *,
    dataset_configs: list,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    core = _unwrap(model)
    core.eval()
    results: dict[str, dict[str, float]] = {}
    aggregate = AccuracyCounts()

    for config in dataset_configs:
        label_bank = load_label_bank(config)
        _, label_to_index, text_bank = _build_text_bank(
            core,
            label_bank,
            batch_size=args.eval_batch_size,
            max_text_length=args.max_text_length,
            device=device,
        )
        collator = LatentSignClipCollator(core.tokenizer, max_text_length=args.max_text_length)
        loader = build_dataloader(
            build_eval_dataset(config, num_frames=args.num_frames),
            collate_fn=collator,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        counts = AccuracyCounts()
        text_bank_device = text_bank.to(device)
        processed = 0
        for batch in loader:
            if batch is None:
                continue
            if args.eval_max_batches and processed >= args.eval_max_batches:
                break
            processed += 1
            valid_positions = [
                i for i, text in enumerate(batch["target_texts"]) if text in label_to_index
            ]
            if not valid_positions:
                continue
            if len(valid_positions) != len(batch["target_texts"]):
                batch["latents"] = batch["latents"][valid_positions]
                batch["num_padding_frames"] = batch["num_padding_frames"][valid_positions]
                batch["target_texts"] = [batch["target_texts"][i] for i in valid_positions]
            batch = _move_batch_to_device(batch, device)
            target_indices = torch.tensor(
                [label_to_index[text] for text in batch["target_texts"]],
                device=device,
                dtype=torch.long,
            )
            video_embeddings = core.encode_video(
                batch["latents"],
                num_padding_frames=batch.get("num_padding_frames"),
            )
            similarities = video_embeddings @ text_bank_device.T
            counts.update(similarities.float().cpu(), target_indices.cpu())

        results[config.name] = counts.as_dict()
        aggregate.top1_correct += counts.top1_correct
        aggregate.top5_correct += counts.top5_correct
        aggregate.total += counts.total

    results["overall"] = aggregate.as_dict()
    core.train()
    return results


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    *,
    wandb_run_id: Optional[str] = None,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"step-{step:07d}.pt"
    payload = {
        "step": step,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "wandb_run_id": wandb_run_id,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("step-*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ddp_enabled, rank, _world, local_rank = _setup_ddp()
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    _seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    wandb_run = None
    if is_main and args.wandb:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prior_run_id = None
        if args.resume:
            prev = find_latest_checkpoint(args.output_dir)
            if prev is not None:
                try:
                    prior_run_id = torch.load(str(prev), map_location="cpu", weights_only=False).get("wandb_run_id")
                except Exception:
                    prior_run_id = None
        try:
            import wandb as _wandb

            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                dir=str(args.output_dir),
                id=prior_run_id,
                resume="allow" if prior_run_id else None,
            )
        except Exception as exc:
            logger.warning("wandb init failed, continuing without: %s", exc)
            wandb_run = None

    dataset_configs = list(DEFAULT_LATENT_DATASET_CONFIGS)
    model = LatentSignCLIPModel(
        text_model_name=args.modernbert,
        max_text_length=args.max_text_length,
        embedding_dim=args.embedding_dim,
        projection_dropout=args.projection_dropout,
        gradient_checkpointing=True,
        num_frames=args.num_frames,
        latent_channels=args.latent_channels,
        latent_size=(args.latent_height, args.latent_width),
        tubelet_size=(args.tubelet_frames, args.tubelet_height, args.tubelet_width),
        vision_embed_dim=args.vision_embed_dim,
        vision_depth=args.vision_depth,
        vision_num_heads=args.vision_num_heads,
        vision_mlp_ratio=args.vision_mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        loss_type=args.loss_type,
    )

    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    collator = LatentSignClipCollator(
        _unwrap(model).tokenizer,
        max_text_length=args.max_text_length,
    )
    train_loader = build_dataloader(
        build_train_dataset(
            dataset_configs,
            num_frames=args.num_frames,
            sample_shuffle=args.sample_shuffle,
            equal_dataset_mix=args.equal_dataset_mix,
        ),
        collate_fn=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    optimizer.zero_grad(set_to_none=True)

    use_grad_scaler = args.mixed_precision == "fp16" and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    accum_steps = max(1, args.gradient_accumulation_steps)
    running_loss = 0.0

    start_step = 0
    resume_path = find_latest_checkpoint(args.output_dir) if args.resume else None
    if resume_path is not None:
        payload = torch.load(str(resume_path), map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        if is_main:
            logger.info("resumed from %s at step=%d", resume_path, start_step)
        if ddp_enabled:
            torch.distributed.barrier()

    for step in range(start_step + 1, args.train_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = _cosine_lr(
                step - 1,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                total_steps=args.train_steps,
            )

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        while batch is None:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
        batch = _move_batch_to_device(batch, device)
        text_features = _split_text_features(batch)

        is_sync_step = step % accum_steps == 0
        sync_ctx = model.no_sync() if (ddp_enabled and not is_sync_step) else contextlib.nullcontext()
        with sync_ctx, _make_autocast(device, args.mixed_precision):
            outputs = model(
                latents=batch["latents"],
                text_features=text_features,
                target_texts=batch.get("target_texts"),
                num_padding_frames=batch.get("num_padding_frames"),
            )
            loss = outputs["loss"] / accum_steps
            if use_grad_scaler:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
        running_loss += float(loss.detach()) * accum_steps

        if is_sync_step:
            if args.grad_clip_norm > 0:
                if use_grad_scaler:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if use_grad_scaler:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if is_main and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            lr = optimizer.param_groups[0]["lr"]
            logger.info("step=%d loss=%.4f lr=%.6e", step, avg_loss, lr)
            if wandb_run is not None:
                wandb_run.log({"train/loss": avg_loss, "train/lr": lr}, step=step)
            running_loss = 0.0

        if step % args.eval_every == 0:
            if ddp_enabled:
                torch.distributed.barrier()
            if is_main:
                metrics = evaluate(model, dataset_configs=dataset_configs, args=args, device=device)
                logger.info("eval step=%d %s", step, json.dumps(metrics, ensure_ascii=False))
                if wandb_run is not None:
                    flat = {
                        f"eval/{ds}/{key}": value
                        for ds, sub in metrics.items()
                        for key, value in sub.items()
                    }
                    wandb_run.log(flat, step=step)
            if ddp_enabled:
                torch.distributed.barrier()

        if step % args.save_every == 0:
            if ddp_enabled:
                torch.distributed.barrier()
            if is_main:
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    step,
                    args,
                    wandb_run_id=(wandb_run.id if wandb_run is not None else None),
                )
                logger.info("saved checkpoint %s", checkpoint_path)
            if ddp_enabled:
                torch.distributed.barrier()

    if is_main:
        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            args.train_steps,
            args,
            wandb_run_id=(wandb_run.id if wandb_run is not None else None),
        )
        logger.info("saved final checkpoint %s", checkpoint_path)
    if wandb_run is not None:
        wandb_run.finish()
    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
