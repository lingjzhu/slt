from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*_object_to_tensor.*")
warnings.filterwarnings("ignore", message=".*_tensor_to_object.*")
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import SignCLIPModel
from .paired import (
    PairedDataConfig,
    PairedSignCollator,
    PairedSignDataset,
    build_paired_dataloader,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiera-ckpt", type=Path, required=True)
    parser.add_argument("--hiera-model-fn", default="hiera_base_128x224")
    parser.add_argument("--modernbert", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sign_clip_paired"))
    parser.add_argument("--base-data-dir", type=Path, default=Path("/mnt/data4"))
    parser.add_argument("--dataset-name", default="all_train_plain_v3")
    parser.add_argument("--manifest", default=None, help="Manifest filename under manifests/")
    parser.add_argument("--languages", default="asl,bsl,csl")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=16.0)
    parser.add_argument("--train-steps", type=int, default=60000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--sampling-rate", type=int, default=1)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=Path("/home/slimelab/Projects/slt/outputs/sign_clip_islr/step-0014000.pt"),
        help="Path to an ISLR sign_clip checkpoint to warm-start from (use '' to disable).",
    )
    parser.add_argument("--loss-type", choices=("sigmoid", "infonce"), default="sigmoid")
    parser.add_argument("--sigmoid-bias-init", type=float, default=-10.0)
    parser.add_argument("--sigmoid-logit-scale-init", type=float, default=math.log(10.0))
    parser.add_argument("--random-horizontal-flip", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--wandb-project", default="sign-clip-paired")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
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
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


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


def save_checkpoint(model, optimizer, step, args, *, wandb_run_id=None) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"step-{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    return path


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("step-*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ddp_enabled, rank, world_size, local_rank = _setup_ddp()
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

    languages = tuple(lang.strip().lower() for lang in args.languages.split(",") if lang.strip())
    data_config = PairedDataConfig(
        base_data_dir=args.base_data_dir,
        dataset_name=args.dataset_name,
        manifest_candidates=(
            (args.manifest,) if args.manifest else
            ("final_train_le16.tsv", "paired_manifest.tsv", "train.tsv")
        ),
        min_duration=args.min_duration,
        max_duration=args.max_duration if args.max_duration > 0 else math.inf,
        allowed_languages=languages if languages else None,
    )

    if is_main:
        logger.info("loading manifest from %s", data_config.resolve_manifest())

    dataset = PairedSignDataset(
        data_config,
        training=True,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        target_fps=args.target_fps,
        crop_size=args.crop_size,
        random_spatial_crop=True,
        random_horizontal_flip=args.random_horizontal_flip,
    )
    if is_main:
        logger.info("PairedSignDataset size=%d", len(dataset))

    model = SignCLIPModel(
        hiera_checkpoint=args.hiera_ckpt,
        hiera_model_fn=args.hiera_model_fn,
        text_model_name=args.modernbert,
        max_text_length=args.max_text_length,
        embedding_dim=args.embedding_dim,
        projection_dropout=args.projection_dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        num_frames=args.num_frames,
        loss_type=args.loss_type,
        sigmoid_bias_init=args.sigmoid_bias_init,
        sigmoid_logit_scale_init=args.sigmoid_logit_scale_init,
    )

    init_from = args.init_from
    init_from_str = str(init_from) if init_from is not None else ""
    if init_from is not None and init_from_str not in ("", ".", "none", "None") and Path(init_from).is_file():
        payload = torch.load(str(init_from), map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        if args.loss_type == "sigmoid":
            state = {k: v for k, v in state.items() if k not in ("logit_scale", "logit_bias")}
        pe_key = "video_backbone.sign_hiera.pos_embed_temporal"
        if pe_key in state:
            target = model.state_dict()[pe_key]
            src = state[pe_key]
            if src.shape != target.shape:
                interp = torch.nn.functional.interpolate(
                    src.transpose(1, 2).float(), size=target.shape[1], mode="linear", align_corners=False,
                ).transpose(1, 2).to(src.dtype)
                state[pe_key] = interp
                if is_main:
                    logger.info("interpolated %s %s -> %s", pe_key, tuple(src.shape), tuple(target.shape))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            logger.info(
                "warm-started from %s (missing=%d unexpected=%d)",
                init_from, len(missing), len(unexpected),
            )
    elif init_from is not None and init_from_str not in ("", ".", "none", "None") and is_main:
        logger.warning("--init-from path not found, skipping: %s", init_from)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.num_frames, args.crop_size, args.crop_size)
        args.pooled_frames = int(model.video_backbone.frame_features(dummy).shape[1])
    if is_main:
        logger.info("SignHiera pooled frames=%d", args.pooled_frames)

    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    collator = PairedSignCollator(
        _unwrap(model).tokenizer,
        max_text_length=args.max_text_length,
        pooled_frames=args.pooled_frames,
    )
    train_loader, sampler = build_paired_dataloader(
        dataset,
        collate_fn=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    use_grad_scaler = args.mixed_precision == "fp16" and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    accum_steps = max(1, args.gradient_accumulation_steps)

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

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    train_iter = iter(train_loader)

    for step in range(start_step + 1, args.train_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = _cosine_lr(
                step - 1,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                total_steps=args.train_steps,
            )

        batch = None
        while batch is None:
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)
        batch = _move_batch_to_device(batch, device)
        text_features = _split_text_features(batch)

        is_sync_step = (step % accum_steps == 0)
        sync_ctx = (
            model.no_sync() if (ddp_enabled and not is_sync_step) else contextlib.nullcontext()
        )
        with sync_ctx, _make_autocast(device, args.mixed_precision):
            outputs = model(
                video=batch["video"],
                video_attention_mask=batch["video_attention_mask"],
                text_features=text_features,
                target_texts=batch.get("target_texts"),
                num_padding_frames=batch.get("video_num_padding_frames"),
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

        if step % args.save_every == 0:
            if ddp_enabled:
                torch.distributed.barrier()
            if is_main:
                path = save_checkpoint(
                    model, optimizer, step, args,
                    wandb_run_id=(wandb_run.id if wandb_run is not None else None),
                )
                logger.info("saved checkpoint %s", path)
            if ddp_enabled:
                torch.distributed.barrier()

    if is_main:
        path = save_checkpoint(
            model, optimizer, args.train_steps, args,
            wandb_run_id=(wandb_run.id if wandb_run is not None else None),
        )
        logger.info("saved final checkpoint %s", path)

    if wandb_run is not None:
        wandb_run.finish()

    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
