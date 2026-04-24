#!/usr/bin/env bash
# Stage-2 V-JEPA-style pretraining: raw mp4 + trainable SignHiera + EMA teacher.
# Run: bash train_stage2_diffusion_vjepa.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST="/mnt/data4/all_train_plain_v3_mae_features_run_001_8fps/manifests/train_feature_detailed.tsv"
MODERNBERT="answerdotai/ModernBERT-large"
HIERA_CKPT="/mnt/data4/mae_pretraining_runs/run_001/checkpoint-00004.pth"
HIERA_MODEL_FN="hiera_base_128x224"

NUM_FRAMES="${NUM_FRAMES:-128}"
SAMPLING_RATE="${SAMPLING_RATE:-1}"
TARGET_FPS="${TARGET_FPS:-8.0}"
CROP_SIZE="${CROP_SIZE:-224}"
POOLED_FRAMES="${POOLED_FRAMES:-}"   # leave empty to auto-probe from SignHiera's actual output shape

BATCH_SIZE="${BATCH_SIZE:-2}"        # raw-video mode is memory-heavy
MAX_STEPS="${MAX_STEPS:-200000}"
LR="${LR:-1e-4}"
EMA_DECAY="${EMA_DECAY:-0.999}"
MASTER_PORT="${MASTER_PORT:-29501}"

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
NUM_WORKERS="${NUM_WORKERS:-$(( CPU_COUNT > NUM_GPUS ? CPU_COUNT / NUM_GPUS : 1 ))}"
if [[ "$NUM_WORKERS" -lt 1 ]]; then
  NUM_WORKERS=1
fi
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPU_COUNT > NUM_GPUS ? CPU_COUNT / NUM_GPUS : 1 ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then
  OMP_NUM_THREADS=1
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ARGS=(
  -m discrete_diffusion_pretraining.train
  --manifest "$MANIFEST"
  --modernbert "$MODERNBERT"
  --hiera-ckpt "$HIERA_CKPT"
  --hiera-model-fn "$HIERA_MODEL_FN"
  --raw-video
  --num-frames "$NUM_FRAMES"
  --sampling-rate "$SAMPLING_RATE"
  --target-fps "$TARGET_FPS"
  --crop-size "$CROP_SIZE"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
  --ema-decay "$EMA_DECAY"
  --num-workers "$NUM_WORKERS"
)
[[ -n "$POOLED_FRAMES" ]] && ARGS+=(--pooled-frames "$POOLED_FRAMES")

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching DDP on $NUM_GPUS GPUs (V-JEPA raw-video)"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    --log-dir /home/slimelab/Projects/slt --redirects 3 "${ARGS[@]}"
else
  echo "Launching single-process V-JEPA raw-video training"
  exec python "${ARGS[@]}"
fi
