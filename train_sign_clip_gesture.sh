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

DATA_ROOT="${DATA_ROOT:-/mnt/data2/sign_language_24fps}"
METADATA_ROOT="${METADATA_ROOT:-$SCRIPT_DIR/islr/webdataset_224}"
MASTER_PORT="${MASTER_PORT:-29512}"

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
  -m sign_clip.train_gesture
  --modernbert "${MODERNBERT:-answerdotai/ModernBERT-base}"
  --output-dir "${OUTPUT_DIR:-$SCRIPT_DIR/outputs/sign_clip_gesture_mixed_128}"
  --data-root "${DATA_ROOT}"
  --metadata-root "${METADATA_ROOT}"
  --train-steps "${TRAIN_STEPS:-80000}"
  --eval-every "${EVAL_EVERY:-1000}"
  --save-every "${SAVE_EVERY:-2000}"
  --batch-size "${BATCH_SIZE:-128}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --lr "${LR:-2e-4}"
  --num-workers "$NUM_WORKERS"
  --feature-dim "${FEATURE_DIM:-1104}"
  --max-frames "${MAX_FRAMES:-256}"
  --gesture-embed-dim "${GESTURE_EMBED_DIM:-512}"
  --gesture-depth "${GESTURE_DEPTH:-6}"
  --gesture-num-heads "${GESTURE_NUM_HEADS:-8}"
  --gesture-mlp-ratio "${GESTURE_MLP_RATIO:-4.0}"
  --max-text-length "${MAX_TEXT_LENGTH:-16}"
  --embedding-dim "${EMBEDDING_DIM:-512}"
  --mixed-precision "${MIXED_PRECISION:-bf16}"
  --csl-val-ratio "${CSL_VAL_RATIO:-0.02}"
  --wandb-project "${WANDB_PROJECT:-sign-clip-gesture}"
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
  echo "Launching DDP sign-clip gesture training on $NUM_GPUS GPUs"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${ARGS[@]}"
else
  echo "Launching single-process sign-clip gesture training"
  exec python "${ARGS[@]}"
fi
