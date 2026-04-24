#!/usr/bin/env bash
# Stage-2 discrete-diffusion pretraining on precomputed pooled features.
# Run: bash train_stage2_diffusion.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MANIFEST="/mnt/data4/all_train_plain_v3_mae_features_run_001_8fps/manifests/train_feature_detailed.tsv"
MODERNBERT="answerdotai/ModernBERT-large"
VISUAL_FEATURE_DIM=768
BATCH_SIZE=32
MAX_STEPS=200000
LR=1e-4
MASTER_PORT=29500

# Precomputed-feature mode: no SignHiera student, no EMA teacher (targets would be
# identical to inputs without a separate teacher feature column).
USE_HIERA=0
USE_TEACHER=0

if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi -L | wc -l)"
else
  NUM_GPUS=1
fi

export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ARGS=(
  -m discrete_diffusion_pretraining.train
  --manifest "$MANIFEST"
  --modernbert "$MODERNBERT"
  --visual-feature-dim "$VISUAL_FEATURE_DIM"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --lr "$LR"
)
[[ "$USE_HIERA" -eq 1 ]] && ARGS+=(--hiera-ckpt /mnt/data4/mae_pretraining_runs/run_001/checkpoint-00004.pth)
[[ "$USE_TEACHER" -eq 0 ]] && ARGS+=(--no-teacher)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching DDP on $NUM_GPUS GPUs"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" --log-dir /tmp/ddp_train_logs --redirects 3 "${ARGS[@]}"
else
  echo "Launching single-process training"
  exec python "${ARGS[@]}"
fi
