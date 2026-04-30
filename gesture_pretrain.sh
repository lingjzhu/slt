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
MASTER_PORT="${MASTER_PORT:-29513}"

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
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPU_COUNT > NUM_GPUS ? CPU_COUNT / NUM_GPUS : 1 ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then OMP_NUM_THREADS=1; fi
export OMP_NUM_THREADS

ARGS=(
  -m gesture_pretraining.run_pretrain
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR:-/mnt/data4/outputs/gesture_pretraining_base}"
  --max-frames "${MAX_FRAMES:-512}"
  --num-workers "$NUM_WORKERS"
  --shuffle-buffer "${SHUFFLE_BUFFER:-2000}"
  --bucket-multiplier "${BUCKET_MULTIPLIER:-4}"
  # Text encoder for contrastive loss (matches sign_clip gesture defaults)
  --text-encoder-kind "${TEXT_ENCODER_KIND:-qwen3-embedding}"
  --text-model-name "${TEXT_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
  --text-attn-implementation "${TEXT_ATTN_IMPL:-flash_attention_2}"
  --max-text-length "${MAX_TEXT_LENGTH:-128}"
  # ModernBERT base architecture (from scratch)
  --feature-dim "${FEATURE_DIM:-1104}"
  --hidden-size "${HIDDEN_SIZE:-768}"
  --num-hidden-layers "${NUM_HIDDEN_LAYERS:-22}"
  --num-attention-heads "${NUM_ATTENTION_HEADS:-12}"
  --intermediate-size "${INTERMEDIATE_SIZE:-1152}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-1024}"
  --global-attn-every-n-layers "${GLOBAL_ATTN_EVERY_N_LAYERS:-3}"
  --local-attention "${LOCAL_ATTENTION:-128}"
  --gesture-attn-implementation "${GESTURE_ATTN_IMPL:-flash_attention_2}"
  --gesture-hidden-activation "${GESTURE_HIDDEN_ACT:-silu}"
  # Losses
  --loss-type "${LOSS_TYPE:-infonce}"
  --temperature "${TEMPERATURE:-0.05}"
  --mask-ratio "${MASK_RATIO:-0.5}"
  --min-span "${MIN_SPAN:-24}"
  --recon-weight "${RECON_WEIGHT:-2.0}"
  # Training
  --batch-size "${BATCH_SIZE:-128}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --train-steps "${TRAIN_STEPS:-80000}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --warmup-steps "${WARMUP_STEPS:-2000}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}"
  --seed "${SEED:-42}"
  # Eval
  --eval-every "${EVAL_EVERY:-2000}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
  --eval-max-samples "${EVAL_MAX_SAMPLES:-20000}"
  # Checkpointing / logging
  --save-every "${SAVE_EVERY:-2000}"
  --log-every "${LOG_EVERY:-20}"
  --wandb-project "${WANDB_PROJECT:-gesture-pretraining}"
)

if [[ -n "${WANDB_RUN_NAME:-}" ]]; then ARGS+=(--wandb-run-name "$WANDB_RUN_NAME"); fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then ARGS+=(--wandb-entity "$WANDB_ENTITY"); fi
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS+=(--resume-from "$RESUME_FROM"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then ARGS+=(--no-resume); fi
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT"); fi
if [[ "${PRETRAINED_STRICT:-0}" == "1" ]]; then ARGS+=(--pretrained-strict); fi
if [[ "${APPLY_LIGER:-1}" == "0" ]]; then ARGS+=(--no-apply-liger); fi
if [[ "${NO_MAE:-0}" == "1" || "${CONTRASTIVE_ONLY:-0}" == "1" || "${SKIP_MAE:-0}" == "1" || "${SKIP_PRETRAINING_STAGE:-0}" == "1" ]]; then
  ARGS+=(--no-mae)
fi
if [[ "${WANDB:-1}" == "0" ]]; then ARGS+=(--no-wandb); fi
if [[ "${COMPILE:-0}" == "1" ]]; then ARGS+=(--compile); fi
if [[ -n "${DATASET_NAMES:-}" ]]; then
  # Pass whitespace-separated dataset names, e.g. DATASET_NAMES="how2sign_24fps csl_daily_24fps"
  # shellcheck disable=SC2206
  DS_ARR=($DATASET_NAMES)
  ARGS+=(--dataset-names "${DS_ARR[@]}")
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching DDP gesture pretraining on $NUM_GPUS GPUs"
  LOG_DIR="${TORCHRUN_LOG_DIR:-${OUTPUT_DIR:-/mnt/data4/outputs/gesture_pretraining_base}/torchrun_logs}"
  mkdir -p "$LOG_DIR"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    --log-dir "$LOG_DIR" --redirects 3 --tee 3 \
    "${ARGS[@]}"
else
  echo "Launching single-process gesture pretraining"
  exec python "${ARGS[@]}"
fi
