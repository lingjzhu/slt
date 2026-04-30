"""Throughput + memory benchmark for full bf16 fine-tuning.

Builds the real model + dataloader, runs a fixed number of optimizer steps,
and prints per-step loss along with summary stats (samples/s, tokens/s,
peak GPU memory).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import build_dataloader, build_train_dataset, discover_shards
from .model import Qwen3SLT
from .train import setup_ddp, seed_everything, unwrap, parameter_count

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="qwen3_slt full bf16 benchmark")
    p.add_argument("--data-root", type=Path, default=Path("/mnt/data2/sign_gestures"))
    p.add_argument("--dataset-names", nargs="+", default=None)
    p.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--gesture-checkpoint", type=Path,
                   default=Path("/mnt/data4/outputs/gesture_pretraining_contrastive_150m/pretrained_encoder"))
    p.add_argument("--max-frames", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--bucket-multiplier", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=5,
                   help="initial steps to skip in throughput averaging (model warmup, autotune)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", type=str, default="bench",
                   help="label printed with the summary line for easy comparison")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    ddp_enabled, rank, world, local_rank = setup_ddp()
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = Qwen3SLT(
        model_name_or_path=args.model_name_or_path,
        gesture_checkpoint=args.gesture_checkpoint,
        unfreeze_encoder=True,
        unfreeze_projector=True,
        unfreeze_decoder=True,
        apply_liger=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, gradient_as_bucket_view=True)
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()

    if is_main:
        total = parameter_count(unwrap(model))
        trainable = parameter_count(unwrap(model), trainable_only=True)
        logger.info("Params: total=%s trainable=%s", f"{total:,}", f"{trainable:,}")

    dataset_names = list(args.dataset_names) if args.dataset_names else None
    shards = discover_shards(args.data_root, split="train", dataset_names=dataset_names)
    if is_main:
        logger.info("Train shards: %d", len(shards))

    bucket_size = args.batch_size * args.bucket_multiplier if args.bucket_multiplier > 0 else 0
    train_dataset = build_train_dataset(
        shards,
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8,
        fused=device.type == "cuda",
    )
    optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_iter = iter(train_loader)
    losses: list[float] = []
    step_times: list[float] = []
    sample_counts: list[int] = []
    label_token_counts: list[int] = []
    seq_lens: list[int] = []

    # Warm up CUDA + iterator
    if is_main:
        logger.info("Warming up for %d steps...", args.warmup_steps)

    for step in range(1, args.steps + args.warmup_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if batch is None:
            continue
        if batch["gesture"].shape[0] < args.batch_size:
            continue

        gesture = batch["gesture"].to(device, non_blocking=True)
        gattn = batch["gesture_attention_mask"].to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.time()

        outputs = model(
            gesture=gesture,
            gesture_attention_mask=gattn,
            sign_languages=batch["sign_languages"],
            output_languages=batch["output_languages"],
            target_texts=batch["captions"],
            max_target_length=args.max_target_length,
        )
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        dt = time.time() - t0

        loss_val = float(loss.detach())
        b = int(gesture.shape[0])
        # Approximate "supervised" tokens per step: average target token count.
        n_label_tokens = sum(
            min(args.max_target_length,
                len(unwrap(model).tokenizer(t, add_special_tokens=False)["input_ids"]) + 1)
            for t in batch["captions"]
        )
        if step > args.warmup_steps:
            losses.append(loss_val)
            step_times.append(dt)
            sample_counts.append(b)
            label_token_counts.append(n_label_tokens)
            seq_lens.append(int(gattn.shape[1]))

        if is_main:
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
            mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
            tag = "warm" if step <= args.warmup_steps else "step"
            logger.info(
                "[%s %s] step=%d loss=%.4f dt=%.3fs T_g=%d B=%d alloc=%.2fGB peak=%.2fGB",
                args.label, tag, step, loss_val, dt, gattn.shape[1], b, mem_alloc, mem_peak,
            )

    if not is_main:
        if ddp_enabled:
            torch.distributed.destroy_process_group()
        return

    # Summary
    n = len(step_times)
    avg_loss = sum(losses) / max(1, n)
    last_loss = losses[-1] if losses else float("nan")
    total_time = sum(step_times)
    avg_step = total_time / max(1, n)
    median_step = sorted(step_times)[n // 2] if n else float("nan")
    samples_per_s = sum(sample_counts) * world / max(total_time, 1e-9)
    label_tok_per_s = sum(label_token_counts) * world / max(total_time, 1e-9)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if torch.cuda.is_available() else 0
    avg_seq_len = sum(seq_lens) / max(1, n)

    logger.info("===== summary [%s] =====", args.label)
    logger.info("steps measured: %d (after %d warmup)", n, args.warmup_steps)
    logger.info("avg loss: %.4f  last loss: %.4f", avg_loss, last_loss)
    logger.info("avg step time: %.3fs  median: %.3fs", avg_step, median_step)
    logger.info("avg gesture seq length: %.1f frames", avg_seq_len)
    logger.info("samples/s (global): %.2f", samples_per_s)
    logger.info("supervised tokens/s (global): %.0f", label_tok_per_s)
    logger.info("peak GPU mem allocated: %.2f GB  reserved: %.2f GB", peak_mem, peak_reserved)

    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
