from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .data.dataset import (
    build_dataloader,
    build_eval_dataset,
    build_train_dataset,
    discover_shards,
)
from .models.pretrain_model import GesturePretrainModel
from .models.text_encoder import FrozenTextEncoder

logger = logging.getLogger(__name__)


# ── DDP setup ─────────────────────────────────────────────────────────────────

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
    return sum(
        p.numel()
        for p in module.parameters()
        if not trainable_only or p.requires_grad
    )


def log_parameter_counts(model: torch.nn.Module, text_encoder: torch.nn.Module) -> None:
    core = unwrap(model)
    total = parameter_count(core)
    trainable = parameter_count(core, trainable_only=True)
    frozen = total - trainable
    logger.info(
        "gesture model parameters: total=%s trainable=%s frozen=%s trainable_pct=%.2f%%",
        f"{total:,}",
        f"{trainable:,}",
        f"{frozen:,}",
        100.0 * trainable / max(1, total),
    )
    for name, module in core.named_children():
        child_total = parameter_count(module)
        if child_total:
            logger.info(
                "  %s parameters: total=%s trainable=%s",
                name,
                f"{child_total:,}",
                f"{parameter_count(module, trainable_only=True):,}",
            )
    logger.info(
        "text encoder parameters: total=%s trainable=%s",
        f"{parameter_count(text_encoder):,}",
        f"{parameter_count(text_encoder, trainable_only=True):,}",
    )


def _swap_layernorms(module: torch.nn.Module, liger_cls: type) -> int:
    """Replace every nn.LayerNorm child with `liger_cls`, copying weight/bias."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.LayerNorm):
            new = liger_cls(
                hidden_size=child.normalized_shape[0],
                eps=child.eps,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None and hasattr(new, "bias") and new.bias is not None:
                    new.bias.copy_(child.bias)
            new.to(child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new)
            n += 1
        else:
            n += _swap_layernorms(child, liger_cls)
    return n


def _swap_modernbert_mlps(module: torch.nn.Module, *, swiglu_cls: type, geglu_cls: type) -> tuple[int, int]:
    """Replace Transformers ModernBERT gated MLPs with Liger fused MLPs."""
    swapped = 0
    skipped = 0
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "ModernBertMLP":
            config = child.config
            activation = getattr(config, "hidden_act", getattr(config, "hidden_activation", "gelu"))
            if not hasattr(config, "hidden_act"):
                config.hidden_act = activation
            if getattr(config, "mlp_dropout", 0.0) != 0.0 or getattr(config, "mlp_bias", False):
                skipped += 1
                continue
            if activation in {"silu", "swish"}:
                new = swiglu_cls(config)
            elif activation in {"gelu", "gelu_pytorch_tanh"}:
                new = geglu_cls(config)
            else:
                skipped += 1
                continue
            with torch.no_grad():
                first, second = child.Wi.weight.chunk(2, dim=0)
                new.gate_proj.weight.copy_(first)
                new.up_proj.weight.copy_(second)
                new.down_proj.weight.copy_(child.Wo.weight)
            new.to(child.Wi.weight.device, dtype=child.Wi.weight.dtype)
            setattr(module, name, new)
            swapped += 1
        else:
            sub_swapped, sub_skipped = _swap_modernbert_mlps(
                child,
                swiglu_cls=swiglu_cls,
                geglu_cls=geglu_cls,
            )
            swapped += sub_swapped
            skipped += sub_skipped
    return swapped, skipped


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, *, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    wandb_run_id: Optional[str] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"step-{step:07d}.pt"
    core = unwrap(model)
    torch.save({
        "step": step,
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "wandb_run_id": wandb_run_id,
    }, path)
    if hasattr(core, "save_pretrained"):
        core.save_pretrained(output_dir / "pretrained_encoder")
        core.save_pretrained(output_dir / f"pretrained_encoder-{step:07d}")
    return path


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("step-*.pt"))
    return candidates[-1] if candidates else None


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool,
    map_location: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    payload = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported pretrained checkpoint format: {checkpoint_path}")
    cleaned = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    return unwrap(model).load_state_dict(cleaned, strict=strict)


# ── Evaluation: retrieval recall@K ───────────────────────────────────────────

@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    *,
    data_root: str | Path,
    max_eval_samples: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_frames: int = 512,
    ks: tuple[int, ...] = (10, 50, 100),
    eval_datasets: tuple[str, ...] = ("how2sign_24fps", "csl_daily_24fps"),
) -> dict[str, float]:
    """
    Computes video→text recall@K on a subset of how2sign/csl_daily by streaming
    directly from the WebDataset shards — no manifest required.
    """
    core = unwrap(model)
    core.eval()
    text_encoder.eval()

    try:
        shards = discover_shards(data_root, split="val", dataset_names=eval_datasets)
    except FileNotFoundError as e:
        logger.warning("No eval shards found: %s", e)
        core.train()
        return {}

    dataset = build_eval_dataset(shards, max_frames=max_frames)
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=min(2, num_workers),
        drop_last=False,
    )

    all_gesture_embs: list[torch.Tensor] = []
    all_captions: list[str] = []
    n_seen = 0
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    for batch in loader:
        if batch is None:
            continue
        gesture = batch["gesture"].to(device)
        attn = batch["gesture_attention_mask"].to(device)
        with amp_ctx:
            emb = core.encode_gesture(gesture, attention_mask=attn).float().cpu()
        all_gesture_embs.append(emb)
        all_captions.extend(batch["captions"])
        n_seen += emb.size(0)
        if max_eval_samples and n_seen >= max_eval_samples:
            break

    if not all_gesture_embs:
        core.train()
        return {}

    gesture_bank = torch.cat(all_gesture_embs, dim=0)  # (N, D)

    # Build unique text bank
    unique_texts = list(dict.fromkeys(all_captions))
    text_embs_list: list[torch.Tensor] = []
    for start in range(0, len(unique_texts), batch_size):
        chunk = unique_texts[start: start + batch_size]
        with amp_ctx:
            emb = text_encoder(chunk).float().cpu()
        text_embs_list.append(emb)
    text_bank = torch.cat(text_embs_list, dim=0)  # (M, D)
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    # Compute similarities (N, M)
    sims = gesture_bank @ text_bank.T

    targets = torch.tensor(
        [text_to_idx[c] for c in all_captions], dtype=torch.long
    )

    metrics: dict[str, float] = {}
    for k in ks:
        topk = sims.topk(min(k, sims.shape[1]), dim=-1).indices  # (N, k)
        hit = (topk == targets.unsqueeze(1)).any(dim=-1).float()
        metrics[f"recall@{k}"] = float(hit.mean())

    core.train()
    return metrics


# ── Main training loop ────────────────────────────────────────────────────────

def train(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ddp_enabled, rank, _world, local_rank = setup_ddp()
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)

    # Discover shards directly — no manifest build, no upfront scanning.
    dataset_names = list(args.dataset_names) if args.dataset_names else None
    train_shards = discover_shards(args.data_root, split="train", dataset_names=dataset_names)
    if is_main:
        logger.info("Train shards: %d across %d datasets",
                    len(train_shards),
                    len({Path(s).parents[1 if Path(s).parent.name in ('train','val','dev','test') else 0].name
                         for s in train_shards}))

    # Frozen text encoder — held OUTSIDE the DDP wrapper so DDP does not
    # traverse its frozen parameters every step. Always eval mode.
    text_encoder = FrozenTextEncoder(
        encoder_kind=args.text_encoder_kind,
        model_name_or_path=args.text_model_name,
        max_text_length=args.max_text_length,
        attn_implementation=args.text_attn_implementation,
        apply_liger=args.apply_liger,
    ).to(device)
    text_encoder.eval()

    # Model
    model = GesturePretrainModel(
        in_dim=args.feature_dim,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        global_attn_every_n_layers=args.global_attn_every_n_layers,
        local_attention=args.local_attention,
        gesture_attn_implementation=args.gesture_attn_implementation,
        gesture_hidden_activation=args.gesture_hidden_activation,
        text_embed_dim=text_encoder.embed_dim,
        temperature=args.temperature,
        loss_type=args.loss_type,
        mask_ratio=args.mask_ratio,
        min_span=args.min_span,
        recon_weight=args.recon_weight,
        no_mae=args.no_mae,
    )
    model = model.to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        # All parameters contribute to every step, so the DDP graph is static
        # — enables DDP bucket optimization. Must be called before first forward.
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()

    if is_main:
        log_parameter_counts(model, text_encoder)

    if args.pretrained_checkpoint is not None:
        info = load_pretrained_weights(
            model,
            Path(args.pretrained_checkpoint),
            strict=args.pretrained_strict,
            map_location=device,
        )
        if is_main:
            logger.info(
                "Warm-started model weights from %s (missing=%d unexpected=%d strict=%s)",
                args.pretrained_checkpoint,
                len(info.missing_keys),
                len(info.unexpected_keys),
                args.pretrained_strict,
            )

    # Liger kernels for ModernBERT blocks while keeping Transformers attention/RoPE.
    if device.type == "cuda":
        try:
            from liger_kernel.transformers import LigerGEGLUMLP, LigerLayerNorm, LigerSwiGLUMLP

            mlp_swapped, mlp_skipped = _swap_modernbert_mlps(
                unwrap(model),
                swiglu_cls=LigerSwiGLUMLP,
                geglu_cls=LigerGEGLUMLP,
            )
            swapped = _swap_layernorms(unwrap(model), LigerLayerNorm)
            if is_main:
                logger.info(
                    "Liger fused ModernBERT MLP swapped in for %d modules (%d skipped)",
                    mlp_swapped,
                    mlp_skipped,
                )
                logger.info("Liger fused LayerNorm swapped in for %d modules", swapped)
        except Exception as e:
            if is_main:
                logger.warning("Liger kernel swaps skipped (%s)", e)
    elif is_main:
        logger.info("Liger CUDA kernel swaps skipped on non-CUDA device")

    # Dataset — streaming WebDataset pipeline with length-bucket batching
    train_dataset = build_train_dataset(
        train_shards,
        max_frames=args.max_frames,
        shuffle_shards=True,
        shuffle_buffer=args.shuffle_buffer,
        length_bucket_size=args.batch_size * args.bucket_multiplier if args.bucket_multiplier > 0 else 0,
        batch_size=args.batch_size,
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    train_iter = iter(train_loader)

    # Optimizer — fused AdamW on CUDA, excludes frozen text encoder parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )
    optimizer.zero_grad(set_to_none=True)
    if is_main:
        logger.info("Optimizer: AdamW (fused=%s) betas=(0.9, 0.95)", use_fused)
        logger.info("Mixed precision: bf16 autocast")

    # W&B
    wandb_run = None
    wandb_run_id = None
    if is_main and args.wandb:
        prior_run_id = None
        if args.resume:
            prev = find_latest_checkpoint(output_dir)
            if prev is not None:
                try:
                    prior_run_id = torch.load(str(prev), map_location="cpu", weights_only=False).get("wandb_run_id")
                except Exception:
                    pass
        try:
            import wandb as _wandb
            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=getattr(args, "wandb_entity", None),
                name=getattr(args, "wandb_run_name", None),
                config=vars(args),
                dir=str(output_dir),
                id=prior_run_id,
                resume="allow" if prior_run_id else None,
            )
            wandb_run_id = wandb_run.id
        except Exception as exc:
            logger.warning("wandb init failed: %s", exc)

    # Resume
    start_step = 0
    resume_path = Path(args.resume_from) if args.resume_from is not None else (
        find_latest_checkpoint(output_dir) if args.resume else None
    )
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        payload = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(payload["model"])
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError as exc:
            if is_main:
                logger.warning("Skipped optimizer state from %s (%s)", resume_path, exc)
        else:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        start_step = int(payload.get("step", 0))
        if is_main:
            logger.info("Resumed from %s at step=%d", resume_path, start_step)
            logger.info("Overrode resumed optimizer LR to %.6e", args.lr)
    if ddp_enabled:
        torch.distributed.barrier()

    accum_steps = max(1, args.gradient_accumulation_steps)
    running = {"loss": 0.0, "loss_contrast": 0.0, "loss_recon": 0.0}

    for step in range(start_step + 1, args.train_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step - 1, base_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=args.train_steps)

        try:
            batch = next(train_iter)
        except StopIteration:
            # WebDataset is an infinite iterable when ResampledShards is used,
            # but this catch is a safety net for deterministic eval pipelines.
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if batch is None:
            continue

        gesture = batch["gesture"].to(device, non_blocking=True)
        attn = batch["gesture_attention_mask"].to(device, non_blocking=True)
        captions = batch["captions"]

        # Compute text embeddings OUTSIDE the DDP module. Frozen, no grad, bf16.
        with torch.no_grad():
            text_emb = text_encoder(captions).to(device)

        is_sync = step % accum_steps == 0
        sync_ctx = model.no_sync() if (ddp_enabled and not is_sync) else contextlib.nullcontext()
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

        with sync_ctx, amp_ctx:
            outputs = model(
                gesture=gesture,
                text_emb=text_emb,
                attention_mask=attn,
                target_texts=captions,
            )
            loss = outputs["loss"] / accum_steps
            loss.backward()

        running["loss"] += float(outputs["loss"].detach())
        running["loss_contrast"] += float(outputs["loss_contrastive"])
        running["loss_recon"] += float(outputs["loss_recon"])

        if is_sync:
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if is_main and step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            log = {k: v / args.log_every for k, v in running.items()}
            logger.info("step=%d loss=%.4f contrast=%.4f recon=%.4f lr=%.2e",
                        step, log["loss"], log["loss_contrast"], log["loss_recon"], lr)
            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss": log["loss"], "train/loss_contrast": log["loss_contrast"],
                     "train/loss_recon": log["loss_recon"], "train/lr": lr},
                    step=step,
                )
            running = {k: 0.0 for k in running}

        if args.eval_every > 0 and step % args.eval_every == 0:
            if ddp_enabled:
                torch.distributed.barrier()
            if is_main:
                metrics = evaluate_retrieval(
                    model,
                    text_encoder,
                    data_root=args.data_root,
                    max_eval_samples=args.eval_max_samples,
                    batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    max_frames=args.max_frames,
                )
                if metrics:
                    logger.info("eval step=%d %s", step, json.dumps(metrics))
                    if wandb_run is not None:
                        wandb_run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            if ddp_enabled:
                torch.distributed.barrier()

        if is_main and step % args.save_every == 0:
            ckpt = save_checkpoint(model, optimizer, step, output_dir, wandb_run_id)
            logger.info("Saved checkpoint %s", ckpt)
        if ddp_enabled and step % args.save_every == 0:
            torch.distributed.barrier()

    if is_main:
        ckpt = save_checkpoint(model, optimizer, args.train_steps, output_dir, wandb_run_id)
        logger.info("Saved final checkpoint %s", ckpt)

    if wandb_run is not None:
        wandb_run.finish()
    if ddp_enabled:
        torch.distributed.destroy_process_group()
