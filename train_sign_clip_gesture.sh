#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default to the slt_qwen conda env (torch 2.5.1 + flash-attn 2 + liger-kernel).
CONDA_ENV="${CONDA_ENV:-slt_qwen}"
CONDA_BIN="${CONDA_BIN:-/home/slimelab/miniconda3/envs/$CONDA_ENV/bin}"
if [[ -d "$CONDA_BIN" ]]; then
  export PATH="$CONDA_BIN:$PATH"
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

DATA_ROOT="${DATA_ROOT:-/mnt/data2/sign_gestures}"
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

TEXT_ENCODER_KIND="${TEXT_ENCODER_KIND:-qwen3-embedding}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
TEXT_ATTN_IMPL="${TEXT_ATTN_IMPL:-flash_attention_2}"
APPLY_LIGER="${APPLY_LIGER:-1}"
FREEZE_TEXT="${FREEZE_TEXT:-1}"
GESTURE_BF16="${GESTURE_BF16:-1}"
CACHED_LOSS="${CACHED_LOSS:-0}"
CACHED_MINI_BATCH_SIZE="${CACHED_MINI_BATCH_SIZE:-32}"

ARGS=(
  -m sign_clip.train_gesture
  --modernbert "${MODERNBERT:-answerdotai/ModernBERT-base}"
  --text-encoder-kind "${TEXT_ENCODER_KIND}"
  --text-model-name "${TEXT_MODEL_NAME}"
  --text-attn-implementation "${TEXT_ATTN_IMPL}"
  --output-dir "${OUTPUT_DIR:-$SCRIPT_DIR/outputs/sign_clip_gesture_qwen3_128}"
  --data-root "${DATA_ROOT}"
  --train-steps "${TRAIN_STEPS:-80000}"
  --eval-every "${EVAL_EVERY:-0}"
  --save-every "${SAVE_EVERY:-2000}"
  --batch-size "${BATCH_SIZE:-128}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --lr "${LR:-2e-4}"
  --num-workers "$NUM_WORKERS"
  --feature-dim "${FEATURE_DIM:-1104}"
  --max-frames "${MAX_FRAMES:-512}"
  --gesture-embed-dim "${GESTURE_EMBED_DIM:-768}"
  --gesture-depth "${GESTURE_DEPTH:-12}"
  --gesture-num-heads "${GESTURE_NUM_HEADS:-12}"
  --gesture-mlp-ratio "${GESTURE_MLP_RATIO:-4.0}"
  --max-text-length "${MAX_TEXT_LENGTH:-128}"
  --mixed-precision "${MIXED_PRECISION:-bf16}"
  --wandb-project "${WANDB_PROJECT:-gesture-pretraining}"
)
if [[ "$APPLY_LIGER" == "0" ]]; then
  ARGS+=(--no-apply-liger)
fi
if [[ "$FREEZE_TEXT" == "1" ]]; then
  ARGS+=(--freeze-text)
fi
if [[ "$GESTURE_BF16" == "1" ]]; then
  ARGS+=(--gesture-bf16)
fi
if [[ "$CACHED_LOSS" == "1" ]]; then
  ARGS+=(--cached-loss --cached-mini-batch-size "$CACHED_MINI_BATCH_SIZE")
fi
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
