from __future__ import annotations

import argparse
import math
import logging
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DEFAULT_DATA_CSV, FULL_DATA_CSV, ISLRDataset, collate_fn, make_label_mapping, split_dataset_frame
from mvit_islr import MViTIsolatedSignModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mvit_v2_s"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated sign classification with torchvision MViT_V2_S.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--view-mode", type=str, choices=("front", "left", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-samples-per-class", type=int, default=1)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"train_mvit_v2_s_{timestamp}.log"

    logger = logging.getLogger("mvit_v2_s_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("Stdout/stderr log file: %s", log_path)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dataframes(args: argparse.Namespace, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.train_csv and args.val_csv:
        train_frame = pd.read_csv(args.train_csv)
        val_frame = pd.read_csv(args.val_csv)
        logger.info("Using explicit train/val CSV files.")
        return train_frame, val_frame

    csv_path = args.csv
    if not csv_path.exists():
        fallback = FULL_DATA_CSV if FULL_DATA_CSV.exists() else None
        if fallback is None:
            raise FileNotFoundError(f"Could not find dataset CSV at {csv_path}.")
        logger.warning("Requested CSV %s was not found. Falling back to %s", csv_path, fallback)
        csv_path = fallback

    train_frame, val_frame = split_dataset_frame(
        csv_path=csv_path,
        val_samples_per_class=args.val_samples_per_class,
        seed=args.seed,
    )
    logger.info(
        "Split %s into %s train rows and %s val rows.",
        csv_path,
        len(train_frame),
        len(val_frame),
    )
    return train_frame, val_frame


def build_loaders(args: argparse.Namespace, logger: logging.Logger) -> tuple[DataLoader, DataLoader, dict[int, int]]:
    train_frame, val_frame = resolve_dataframes(args, logger)
    label_to_index = make_label_mapping([train_frame, val_frame])

    train_dataset = ISLRDataset(
        dataframe=train_frame,
        training=True,
        label_to_index=label_to_index,
        view_mode=args.view_mode,
    )
    val_dataset = ISLRDataset(
        dataframe=val_frame,
        training=False,
        label_to_index=label_to_index,
        view_mode=args.view_mode,
    )

    logger.info(
        "Training samples: %s | Validation samples: %s | Classes: %s",
        len(train_dataset),
        len(val_dataset),
        len(label_to_index),
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, label_to_index


def move_videos_to_device(videos: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [video.to(device, non_blocking=True) for video in videos]


def get_num_batches(loader: DataLoader, max_batches: int | None) -> int:
    return min(len(loader), max_batches) if max_batches is not None else len(loader)


def build_lr_scheduler(
    total_optimizer_steps: int,
    warmup_ratio: float,
    min_learning_rate: float,
    base_learning_rate: float,
) -> Callable[[int], float]:
    if warmup_ratio > 0.0:
        warmup_steps = min(total_optimizer_steps, max(1, math.ceil(total_optimizer_steps * warmup_ratio)))
    else:
        warmup_steps = 0
    min_lr_scale = min_learning_rate / base_learning_rate if base_learning_rate > 0 else 0.0

    def get_learning_rate(current_step: int) -> float:
        if total_optimizer_steps <= 1:
            return base_learning_rate

        if warmup_steps > 0 and current_step < warmup_steps:
            scale = float(current_step + 1) / float(warmup_steps)
            return base_learning_rate * scale

        decay_steps = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr_scale + (1.0 - min_lr_scale) * cosine_scale
        return base_learning_rate * scale

    return get_learning_rate


def run_epoch(
    model: MViTIsolatedSignModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Callable[[int], float] | None,
    scaler: torch.amp.GradScaler | None,
    logger: logging.Logger,
    max_batches: int | None,
    log_prefix: str,
    accumulation_steps: int = 1,
    optimizer_step_offset: int = 0,
) -> tuple[float, float, float, int]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    autocast_enabled = device.type == "cuda"
    steps_run = 0
    optimizer_updates = 0

    if is_training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    for step, (videos, labels) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break

        steps_run = step

        labels = labels.to(device, non_blocking=True)
        videos = move_videos_to_device(videos, device)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(videos)
                loss = criterion(logits, labels)
                scaled_loss = loss / accumulation_steps

            if is_training:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = step % accumulation_steps == 0
                if max_batches is not None:
                    should_step = should_step or step == min(len(loader), max_batches)
                else:
                    should_step = should_step or step == len(loader)

                if should_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        next_learning_rate = scheduler(optimizer_step_offset + optimizer_updates + 1)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = next_learning_rate
                    optimizer_updates += 1
                    optimizer.zero_grad(set_to_none=True)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

        if step == 1 or step % 10 == 0:
            logger.info(
                "%s step %s/%s | loss %.4f | acc %.2f%%",
                log_prefix,
                step,
                len(loader),
                total_loss / max(total_examples, 1),
                100.0 * total_correct / max(total_examples, 1),
            )

    if is_training and steps_run == 0:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    average_loss = total_loss / max(total_examples, 1)
    average_accuracy = 100.0 * total_correct / max(total_examples, 1)
    current_learning_rate = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    return average_loss, average_accuracy, current_learning_rate, optimizer_updates


def save_checkpoint(
    checkpoint_dir: Path,
    model: MViTIsolatedSignModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val_accuracy: float,
    completed_optimizer_steps: int,
    label_to_index: dict[int, int],
    args: argparse.Namespace,
    filename: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    serialized_args = {}
    for key, value in vars(args).items():
        serialized_args[key] = str(value) if isinstance(value, Path) else value

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_accuracy": best_val_accuracy,
            "completed_optimizer_steps": completed_optimizer_steps,
            "label_to_index": label_to_index,
            "train_args": serialized_args,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: MViTIsolatedSignModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    logger: logging.Logger,
    device: torch.device,
) -> tuple[int, float, int, dict[int, int] | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    resume_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_accuracy = float(checkpoint.get("best_val_accuracy", -1.0))
    completed_optimizer_steps = int(checkpoint.get("completed_optimizer_steps", 0))
    checkpoint_labels = checkpoint.get("label_to_index")

    logger.info(
        "Resumed checkpoint %s from epoch %s with best val acc %.2f%% and %s optimizer steps completed.",
        checkpoint_path,
        resume_epoch,
        best_val_accuracy,
        completed_optimizer_steps,
    )
    return resume_epoch, best_val_accuracy, completed_optimizer_steps, checkpoint_labels


def main() -> None:
    args = parse_args()
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be at least 1.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in the range [0.0, 1.0).")
    if args.min_learning_rate < 0.0:
        raise ValueError("--min-learning-rate must be non-negative.")

    logger = setup_logging(args.log_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Arguments: %s", vars(args))
    logger.info(
        "Effective video batch size: %s",
        args.batch_size * args.accumulation_steps,
    )

    train_loader, val_loader, label_to_index = build_loaders(args, logger)

    model = MViTIsolatedSignModel(
        num_classes=len(label_to_index),
        pretrained=not args.no_pretrained,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        segment_batch_size=args.segment_batch_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_batches_per_epoch = get_num_batches(train_loader, args.max_train_batches)
    optimizer_steps_per_epoch = math.ceil(train_batches_per_epoch / args.accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = build_lr_scheduler(
        total_optimizer_steps=total_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
        min_learning_rate=args.min_learning_rate,
        base_learning_rate=args.learning_rate,
    )
    initial_learning_rate = scheduler(0)
    for param_group in optimizer.param_groups:
        param_group["lr"] = initial_learning_rate
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    logger.info(
        "LR schedule: warmup_ratio=%.3f | total_optimizer_steps=%s | min_lr=%.8f | initial_lr=%.8f",
        args.warmup_ratio,
        total_optimizer_steps,
        args.min_learning_rate,
        initial_learning_rate,
    )

    start_epoch = 1
    best_val_accuracy = -1.0
    completed_optimizer_steps = 0

    resume_path = args.resume
    if resume_path is not None:
        if str(resume_path).lower() == "last":
            resume_path = args.checkpoint_dir / "last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        (
            start_epoch,
            best_val_accuracy,
            completed_optimizer_steps,
            checkpoint_label_to_index,
        ) = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            logger=logger,
            device=device,
        )
        if checkpoint_label_to_index is not None and checkpoint_label_to_index != label_to_index:
            raise ValueError("Checkpoint label mapping does not match the current dataset split.")

        resumed_learning_rate = scheduler(completed_optimizer_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = resumed_learning_rate
        logger.info("Restored scheduled learning rate to %.8f", resumed_learning_rate)

    if start_epoch > args.epochs:
        logger.info(
            "Checkpoint is already at epoch %s, which is past requested --epochs=%s. Nothing to do.",
            start_epoch - 1,
            args.epochs,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss, train_accuracy, current_learning_rate, optimizer_updates = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            logger=logger,
            max_batches=args.max_train_batches,
            log_prefix=f"train epoch {epoch}",
            accumulation_steps=args.accumulation_steps,
            optimizer_step_offset=completed_optimizer_steps,
        )
        completed_optimizer_steps += optimizer_updates
        val_loss, val_accuracy, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scheduler=None,
            scaler=None,
            logger=logger,
            max_batches=args.max_val_batches,
            log_prefix=f"val epoch {epoch}",
            accumulation_steps=1,
        )

        logger.info(
            "epoch %s done | train loss %.4f | train acc %.2f%% | val loss %.4f | val acc %.2f%% | lr %.6f | %.1fs",
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            current_learning_rate,
            time.time() - start,
        )

        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_accuracy=max(best_val_accuracy, val_accuracy),
            completed_optimizer_steps=completed_optimizer_steps,
            label_to_index=label_to_index,
            args=args,
            filename="last.pt",
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                completed_optimizer_steps=completed_optimizer_steps,
                label_to_index=label_to_index,
                args=args,
                filename="best.pt",
            )
            logger.info("Saved new best checkpoint with val acc %.2f%%", val_accuracy)

    logger.info("Training finished. Best val accuracy: %.2f%%", best_val_accuracy)


if __name__ == "__main__":
    main()
