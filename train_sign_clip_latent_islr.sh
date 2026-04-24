#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SIGN_CLIP_LATENT_WEBDS_ROOT="${SIGN_CLIP_LATENT_WEBDS_ROOT:-$SCRIPT_DIR/islr/webdataset_224_leanvae_latents}"

MASTER_PORT="${MASTER_PORT:-29503}"

if command -v python >/dev/null 2>&1; then
  NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')}"
else
  NUM_GPUS="${NUM_GPUS:-1}"
fi

if command -v nproc >/dev/null 2>&1; then
  CPU_COUNT="$(nproc)"
else
  CPU_COUNT=8
fi
NUM_WORKERS="${NUM_WORKERS:-8}"
if [[ "$NUM_WORKERS" -lt 1 ]]; then
  NUM_WORKERS=1
fi
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPU_COUNT > NUM_GPUS ? CPU_COUNT / NUM_GPUS : 1 ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then
  OMP_NUM_THREADS=1
fi
export OMP_NUM_THREADS

ARGS=(
  -m sign_clip.train_latent
  --modernbert "${MODERNBERT:-answerdotai/ModernBERT-base}"
  --output-dir "${OUTPUT_DIR:-$SCRIPT_DIR/outputs/sign_clip_latent_islr}"
  --train-steps "${TRAIN_STEPS:-40000}"
  --eval-every "${EVAL_EVERY:-20000}"
  --save-every "${SAVE_EVERY:-2000}"
  --batch-size "${BATCH_SIZE:-64}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --lr "${LR:-2e-4}"
  --num-workers "$NUM_WORKERS"
  --num-frames "${NUM_FRAMES:-80}"
  --vision-embed-dim "${VISION_EMBED_DIM:-768}"
  --vision-depth "${VISION_DEPTH:-12}"
  --vision-num-heads "${VISION_NUM_HEADS:-12}"
  --tubelet-frames "${TUBELET_FRAMES:-1}"
  --tubelet-height "${TUBELET_HEIGHT:-7}"
  --tubelet-width "${TUBELET_WIDTH:-7}"
  --max-text-length "${MAX_TEXT_LENGTH:-16}"
  --mixed-precision "${MIXED_PRECISION:-bf16}"
  --equal-dataset-mix
  --wandb-project "${WANDB_PROJECT:-sign-clip-latent-islr}"
)
if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ "${WANDB:-1}" == "0" ]]; then
  ARGS+=(--no-wandb)
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching DDP latent sign-clip training on $NUM_GPUS GPUs"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${ARGS[@]}"
else
  echo "Launching single-process latent sign-clip training"
  exec python "${ARGS[@]}"
fi
