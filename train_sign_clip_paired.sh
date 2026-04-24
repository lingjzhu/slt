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

HIERA_CKPT="${HIERA_CKPT:-/mnt/data4/mae_pretraining_runs/run_001/checkpoint-00004.pth}"
if [[ -z "${HIERA_CKPT}" ]]; then
  echo "Set HIERA_CKPT=/path/to/signhiera_checkpoint.pt" >&2
  exit 1
fi

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
  -m sign_clip.train_paired
  --hiera-ckpt "$HIERA_CKPT"
  --modernbert "${MODERNBERT:-answerdotai/ModernBERT-base}"
  --output-dir "${OUTPUT_DIR:-$SCRIPT_DIR/outputs/sign_clip_paired}"
  --base-data-dir "${BASE_DATA_DIR:-/mnt/data4}"
  --dataset-name "${DATASET_NAME:-all_train_plain_v3}"
  --languages "${LANGUAGES:-asl,bsl,csl}"
  --max-duration "${MAX_DURATION:-16.0}"
  --train-steps "${TRAIN_STEPS:-200000}"
  --save-every "${SAVE_EVERY:-2000}"
  --batch-size "${BATCH_SIZE:-20}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-2}"
  --lr "${LR:-2e-4}"
  --warmup-steps "${WARMUP_STEPS:-2000}"
  --num-workers "$NUM_WORKERS"
  --num-frames "${NUM_FRAMES:-128}"
  --target-fps "${TARGET_FPS:-8}"
  --crop-size "${CROP_SIZE:-224}"
  --max-text-length "${MAX_TEXT_LENGTH:-96}"
  --mixed-precision "${MIXED_PRECISION:-bf16}"
  --wandb-project "${WANDB_PROJECT:-sign-clip-paired}"
  --loss-type "${LOSS_TYPE:-sigmoid}"
)
if [[ -n "${INIT_FROM+x}" ]]; then
  ARGS+=(--init-from "$INIT_FROM")
fi
if [[ -n "${MANIFEST:-}" ]]; then
  ARGS+=(--manifest "$MANIFEST")
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
if [[ "${RANDOM_HORIZONTAL_FLIP:-0}" == "1" ]]; then
  ARGS+=(--random-horizontal-flip)
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching DDP sign-clip paired training on $NUM_GPUS GPUs"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${ARGS[@]}" "$@"
else
  echo "Launching single-process sign-clip paired training"
  exec python "${ARGS[@]}" "$@"
fi
