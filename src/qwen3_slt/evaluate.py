"""Standalone evaluation for Qwen3 SLT checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from .model import Qwen3SLT
from .train import evaluate_split, find_latest_checkpoint, seed_everything
from t5_slt.metrics import save_json

logger = logging.getLogger(__name__)


def _load_run_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get(config: dict[str, Any], name: str, default: Any) -> Any:
    value = config.get(name, default)
    if value is None:
        return default
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Qwen3 SLT checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=Path("/mnt/data4/outputs/qwen3_slt_stage2_full"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--dataset-names", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--num-beams", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--no-repeat-ngram-size", type=int, default=None)
    p.add_argument("--length-penalty", type=float, default=None)
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = parse_args()
    config = _load_run_config(args.run_dir / "run_config.json")

    checkpoint = args.checkpoint or find_latest_checkpoint(args.run_dir)
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint found in {args.run_dir}")

    seed_everything(args.seed if args.seed is not None else int(_get(config, "seed", 42)))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_root = args.data_root or Path(_get(config, "data_root", "/mnt/data2/sign_gestures"))
    output_dir = args.output_dir or (args.run_dir / "test_eval" / checkpoint.stem / "_".join(args.dataset_names))

    logger.info("Loading checkpoint %s", checkpoint)
    model = Qwen3SLT(
        model_name_or_path=str(_get(config, "model_name_or_path", "Qwen/Qwen3-0.6B-Base")),
        gesture_in_dim=int(_get(config, "gesture_in_dim", 1104)),
        gesture_hidden_size=int(_get(config, "gesture_hidden_size", 768)),
        gesture_num_hidden_layers=int(_get(config, "gesture_num_hidden_layers", 22)),
        gesture_num_attention_heads=int(_get(config, "gesture_num_attention_heads", 12)),
        gesture_intermediate_size=int(_get(config, "gesture_intermediate_size", 1152)),
        gesture_max_position_embeddings=int(_get(config, "gesture_max_position_embeddings", 1024)),
        gesture_global_attn_every_n_layers=int(_get(config, "gesture_global_attn_every_n_layers", 3)),
        gesture_local_attention=int(_get(config, "gesture_local_attention", 128)),
        gesture_attn_implementation=str(_get(config, "gesture_attn_impl", "flash_attention_2")),
        projection_hidden_dim=int(_get(config, "projection_hidden_dim", 0)) or None,
        projection_dropout=float(_get(config, "projection_dropout", 0.1)),
        attn_implementation=str(_get(config, "attn_impl", "flash_attention_2")),
        torch_dtype=torch.bfloat16,
        gesture_checkpoint=Path(_get(config, "gesture_checkpoint", "/mnt/data4/outputs/gesture_pretraining_contrastive_150m/pretrained_encoder")),
        unfreeze_encoder=False,
        unfreeze_projector=False,
        unfreeze_decoder=False,
        apply_liger=bool(_get(config, "apply_liger", True)),
    ).to(device)

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()

    save_json(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else None,
            "run_dir": str(args.run_dir),
            "data_root": str(data_root),
            "dataset_names": args.dataset_names,
            "split": args.split,
            "device": str(device),
        },
        output_dir / "eval_config.json",
    )

    metrics = evaluate_split(
        split_name=args.split,
        model=model,
        data_root=data_root,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size or int(_get(config, "eval_batch_size", 4)),
        num_workers=args.num_workers if args.num_workers is not None else int(_get(config, "num_workers", 6)),
        max_frames=args.max_frames or int(_get(config, "max_frames", 512)),
        max_new_tokens=args.max_new_tokens or int(_get(config, "eval_max_new_tokens", 128)),
        num_beams=args.num_beams or int(_get(config, "eval_num_beams", 4)),
        repetition_penalty=args.repetition_penalty if args.repetition_penalty is not None else float(_get(config, "eval_repetition_penalty", 1.0)),
        no_repeat_ngram_size=args.no_repeat_ngram_size if args.no_repeat_ngram_size is not None else int(_get(config, "eval_no_repeat_ngram_size", 0)),
        length_penalty=args.length_penalty if args.length_penalty is not None else float(_get(config, "eval_length_penalty", 1.0)),
        early_stopping=args.early_stopping or bool(_get(config, "eval_early_stopping", False)),
        max_eval_samples=args.max_eval_samples,
        dataset_names=tuple(args.dataset_names),
    )
    save_json(metrics, output_dir / f"{args.split}_metrics.json")
    logger.info("Saved predictions and metrics to %s", output_dir)


if __name__ == "__main__":
    main()
