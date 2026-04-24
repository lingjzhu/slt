from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from .data import (
    DiffusionExample,
    DiscreteDiffusionCollator,
    DiscreteDiffusionDataset,
)
from .metrics import TrainingMetrics
from .model import DiscreteDiffusionModel
from .raw_video_data import (
    RawVideoCollator,
    RawVideoDataset,
    load_raw_manifest,
)
from .sign_hiera_backbone import build_sign_hiera_student
from .teacher import EMATeacher


logger = logging.getLogger(__name__)


_LANGUAGE_NAME_MAP = {
    "asl": "English",
    "bsl": "English",
    "csl": "Chinese",
    "en": "English",
    "zh": "Chinese",
}


def _resolve_language_name(code: str) -> str:
    code = (code or "").strip().lower()
    return _LANGUAGE_NAME_MAP.get(code, code.capitalize() or "English")


def _resolve_feature_path(manifest_path: Path, raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((manifest_path.parent.parent / candidate).resolve())


def _load_manifest(manifest_path: Path, prompt_template: str) -> list[DiffusionExample]:
    import csv

    manifest_path = Path(manifest_path)
    examples: list[DiffusionExample] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            lang_code = row.get("language", "")
            language_name = _resolve_language_name(lang_code)
            teacher_raw = row.get("teacher_feature_path") or None
            teacher_resolved = (
                _resolve_feature_path(manifest_path, teacher_raw) if teacher_raw else None
            )
            examples.append(
                DiffusionExample(
                    sample_id=Path(row["video_path"]).stem,
                    feature_path=_resolve_feature_path(manifest_path, row["feature_path"]),
                    teacher_feature_path=teacher_resolved,
                    prompt_text=prompt_template.format(language=language_name),
                    target_text=row.get("caption", ""),
                    language=lang_code,
                )
            )
    if not examples:
        raise ValueError(f"No examples in {manifest_path}")
    return examples


def _setup_ddp() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world
        )
        return True, rank, world, local
    return False, 0, 1, 0


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def train_loop(
    model: nn.Module,
    teacher: Optional[EMATeacher],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_steps: int,
    is_main: bool,
    log_every: int = 10,
) -> TrainingMetrics:
    model.train()
    core = _unwrap(model)
    metrics = TrainingMetrics()
    step = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        if "visual_raw" in batch:
            visual_raw = batch["visual_raw"]
            fwd_kwargs = {"visual_raw": visual_raw}
            if teacher is not None:
                with torch.no_grad():
                    visual_targets = teacher(visual_raw)
            else:
                raise RuntimeError("raw-video mode requires an EMA teacher")
        else:
            visual_raw = batch["visual_features"]
            if teacher is not None and core.student_backbone is not None:
                with torch.no_grad():
                    visual_targets = teacher(visual_raw)
            else:
                visual_targets = batch["visual_targets"]
            if core.student_backbone is not None:
                fwd_kwargs = {"visual_raw": visual_raw}
            else:
                fwd_kwargs = {"visual_features": visual_raw}

        out = model(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            text_labels=batch["text_labels"],
            text_mask_a=batch["text_mask_a"],
            text_mask_b=batch["text_mask_b"],
            visual_attention_mask=batch["visual_attention_mask"],
            visual_targets=visual_targets,
            visual_mask_a=batch["visual_mask_a"],
            visual_mask_b=batch["visual_mask_b"],
            **fwd_kwargs,
        )
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if teacher is not None and core.student_backbone is not None:
            teacher.update(core.student_backbone)

        metrics.update_from_step(float(loss), out["pass_a"], out["pass_b"])
        step += 1
        if is_main and step % log_every == 0:
            logger.info("step=%d %s", step, metrics.as_dict())
        if step >= max_steps:
            break
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modernbert", default="answerdotai/ModernBERT-base")
    parser.add_argument("--hiera-ckpt", type=Path, default=None,
                        help="Path to SignHiera MAE checkpoint (encoder weights will be loaded).")
    parser.add_argument("--hiera-model-fn", default="hiera_base_128x224")
    parser.add_argument("--visual-feature-dim", type=int, default=None,
                        help="If omitted and --hiera-ckpt is given, inferred from the backbone.")
    parser.add_argument("--teacher-feature-dim", type=int, default=None)
    parser.add_argument("--prompt-template", default="translate into {language}:")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-video", action="store_true",
                        help="Enable V-JEPA mode: decode mp4 on the fly, run trainable SignHiera student, use EMA teacher.")
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--sampling-rate", type=int, default=1)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--pooled-frames", type=int, default=None,
                        help="T_out from SignHiera; defaults to num_frames // 2 for hiera_base_128x224.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)

    ddp_enabled, rank, world, local_rank = _setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.modernbert)

    # Build student backbone first so --raw-video can probe pooled length.
    student_backbone: Optional[nn.Module] = None
    visual_feature_dim = args.visual_feature_dim
    if args.hiera_ckpt is not None:
        student_backbone = build_sign_hiera_student(
            args.hiera_ckpt, model_fn=args.hiera_model_fn
        )
        inferred_dim = student_backbone.feature_dim
        visual_feature_dim = visual_feature_dim or inferred_dim
        if is_main:
            logger.info("Loaded SignHiera from %s, feature_dim=%d", args.hiera_ckpt, inferred_dim)
    if visual_feature_dim is None:
        raise ValueError("--visual-feature-dim required when --hiera-ckpt is not provided")

    if args.raw_video:
        if student_backbone is None:
            raise ValueError("--raw-video requires --hiera-ckpt")
        raw_examples = load_raw_manifest(args.manifest)
        dataset = RawVideoDataset(
            raw_examples,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            target_fps=args.target_fps,
            crop_size=args.crop_size,
        )
        if args.pooled_frames is not None:
            pooled_frames = args.pooled_frames
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, args.num_frames, args.crop_size, args.crop_size)
                probe_out = student_backbone(dummy)
            pooled_frames = int(probe_out.shape[1])
            if is_main:
                logger.info("Probed pooled_frames=%d from SignHiera output", pooled_frames)
        collator = RawVideoCollator(
            tokenizer,
            prompt_template=args.prompt_template,
            pooled_frames=pooled_frames,
        )
    else:
        examples = _load_manifest(args.manifest, args.prompt_template)
        dataset = DiscreteDiffusionDataset(examples)
        collator = DiscreteDiffusionCollator(tokenizer)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if ddp_enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DiscreteDiffusionModel(
        modernbert_name_or_path=args.modernbert,
        visual_feature_dim=visual_feature_dim,
        teacher_feature_dim=args.teacher_feature_dim or visual_feature_dim,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        student_backbone=student_backbone,
    ).to(device)

    teacher: Optional[EMATeacher] = None
    if not args.no_teacher and student_backbone is not None:
        teacher = EMATeacher(student_backbone, decay=args.ema_decay).to(device)

    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    train_loop(
        model, teacher, loader, optimizer,
        device=device, max_steps=args.max_steps, is_main=is_main,
    )

    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
