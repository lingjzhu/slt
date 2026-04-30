from __future__ import annotations

import argparse
from pathlib import Path

from .trainer import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture pretraining with InfoNCE + masked reconstruction")

    # Data
    p.add_argument("--data-root", type=Path, default=Path("/mnt/data2/sign_gestures"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gesture_pretraining"))
    p.add_argument("--dataset-names", nargs="+", default=None,
                   help="Restrict which gesture datasets to include (e.g. how2sign_24fps csl_daily_24fps). "
                        "Default streams from all datasets.")
    p.add_argument("--max-frames", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shuffle-buffer", type=int, default=2000,
                   help="Per-worker shuffle buffer size. 0 disables shuffling.")
    p.add_argument("--bucket-multiplier", type=int, default=4,
                   help="Length-bucket buffer size = bucket_multiplier * batch_size. "
                        "Bigger → less padding waste but less length diversity per step. "
                        "0 disables length bucketing.")
    p.add_argument(
        "--text-encoder-kind",
        choices=("jina", "qwen3-embedding"),
        default="qwen3-embedding",
    )
    p.add_argument(
        "--text-model-name",
        default=None,
        help="Override text model. Defaults to Qwen/Qwen3-Embedding-0.6B for qwen3-embedding "
             "or jinaai/jina-embeddings-v5-text-nano for jina.",
    )
    p.add_argument("--max-text-length", type=int, default=32)
    p.add_argument("--text-attn-implementation", default="flash_attention_2")
    p.add_argument("--apply-liger", dest="apply_liger", action="store_true", default=True)
    p.add_argument("--no-apply-liger", dest="apply_liger", action="store_false")

    # Model — gesture encoder (ModernBERT base)
    p.add_argument("--feature-dim", type=int, default=1104)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-hidden-layers", type=int, default=22)
    p.add_argument("--num-attention-heads", type=int, default=12)
    p.add_argument("--intermediate-size", type=int, default=1152)
    p.add_argument("--max-position-embeddings", type=int, default=8192)
    p.add_argument("--global-attn-every-n-layers", type=int, default=3)
    p.add_argument("--local-attention", type=int, default=128)
    p.add_argument(
        "--gesture-attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
        help="Attention backend for the ModernBERT gesture encoder.",
    )
    p.add_argument(
        "--gesture-hidden-activation",
        choices=("silu", "swish", "gelu", "gelu_pytorch_tanh"),
        default="silu",
        help="Gated MLP activation for the ModernBERT gesture encoder. "
             "silu/swish enables Liger SwiGLU patching.",
    )

    # Loss
    p.add_argument("--loss-type", choices=("infonce", "sigmoid"), default="infonce")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--min-span", type=int, default=24,
                   help="Minimum masked span in original frame units")
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--no-mae", action="store_true",
                   help="Disable masked input and reconstruction loss; train contrastive loss only")

    # Training
    p.add_argument("--batch-size", type=int, default=112)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")

    # Eval
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-max-samples", type=int, default=2000,
                   help="Cap the eval set to this many samples for speed")

    # Checkpointing / logging
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume from a specific checkpoint instead of the latest checkpoint")
    p.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Warm-start model weights from a checkpoint, but start optimizer/step fresh.",
    )
    p.add_argument(
        "--pretrained-strict",
        action="store_true",
        help="Require all pretrained checkpoint keys to match exactly.",
    )
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="gesture-pretraining")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
