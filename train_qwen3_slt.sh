#!/usr/bin/env bash
# Train a multimodal Qwen3 model for sign language translation.
#
# Encoder: pretrained GestureEncoder (ModernBERT base, 2x temporal downsample).
# Decoder: Qwen3-0.6B (causal LM) with FA2 + Liger kernels (FLCE, RMSNorm, RoPE, SwiGLU).
# Sequence: [<|im_start|>user\\n] [gesture features] [\\nTranslate <sign> to <lang><|im_end|>\\n<|im_start|>assistant\\n] [target] [<|im_end|>]
#
# Override unfreeze toggles to pick a training setting:
#   UNFREEZE_PROJECTOR=1                            (default: just the projector)
#   UNFREEZE_ENCODER=1 UNFREEZE_PROJECTOR=1         (encoder + projector)
#   UNFREEZE_PROJECTOR=1 UNFREEZE_DECODER=1         (projector + decoder)
#   UNFREEZE_ENCODER=1 UNFREEZE_PROJECTOR=1 UNFREEZE_DECODER=1   (full unfreeze)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data4/outputs/qwen3_slt}"
GESTURE_CHECKPOINT="${GESTURE_CHECKPOINT:-/mnt/data4/outputs/gesture_pretraining_contrastive_150m/pretrained_encoder}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
MASTER_PORT="${MASTER_PORT:-29515}"

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
NUM_WORKERS="${NUM_WORKERS:-6}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPU_COUNT > NUM_GPUS ? CPU_COUNT / NUM_GPUS : 1 ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then OMP_NUM_THREADS=1; fi
export OMP_NUM_THREADS

# Training hyperparams
MAX_FRAMES="${MAX_FRAMES:-512}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PROJECTION_HIDDEN_DIM="${PROJECTION_HIDDEN_DIM:-0}"
PROJECTION_DROPOUT="${PROJECTION_DROPOUT:-0.1}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-2000}"
BUCKET_MULTIPLIER="${BUCKET_MULTIPLIER:-4}"

# Eval
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2000}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.0}"
EVAL_NO_REPEAT_NGRAM_SIZE="${EVAL_NO_REPEAT_NGRAM_SIZE:-0}"
EVAL_LENGTH_PENALTY="${EVAL_LENGTH_PENALTY:-1.0}"
EVAL_DATASETS="${EVAL_DATASETS:-how2sign_24fps csl_daily_24fps}"

# Checkpointing / logging
SAVE_EVERY="${SAVE_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-20}"
SEED="${SEED:-42}"

# Freeze toggles (default: train only the projector)
UNFREEZE_ENCODER="${UNFREEZE_ENCODER:-0}"
UNFREEZE_PROJECTOR="${UNFREEZE_PROJECTOR:-1}"
UNFREEZE_DECODER="${UNFREEZE_DECODER:-0}"

ARGS=(
  -m qwen3_slt.train
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --model-name-or-path "${MODEL_NAME}"
  --gesture-checkpoint "${GESTURE_CHECKPOINT}"
  --max-frames "${MAX_FRAMES}"
  --max-target-length "${MAX_TARGET_LENGTH}"
  --num-workers "${NUM_WORKERS}"
  --shuffle-buffer "${SHUFFLE_BUFFER}"
  --bucket-multiplier "${BUCKET_MULTIPLIER}"
  --projection-hidden-dim "${PROJECTION_HIDDEN_DIM}"
  --projection-dropout "${PROJECTION_DROPOUT}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --train-steps "${TRAIN_STEPS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-steps "${WARMUP_STEPS}"
  --grad-clip-norm "${GRAD_CLIP}"
  --eval-every "${EVAL_EVERY}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-num-beams "${EVAL_NUM_BEAMS}"
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  --eval-repetition-penalty "${EVAL_REPETITION_PENALTY}"
  --eval-no-repeat-ngram-size "${EVAL_NO_REPEAT_NGRAM_SIZE}"
  --eval-length-penalty "${EVAL_LENGTH_PENALTY}"
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
  --wandb-project "${WANDB_PROJECT:-qwen3-slt}"
)

if [[ "${UNFREEZE_ENCODER}" == "1" ]]; then ARGS+=(--unfreeze-encoder); fi
if [[ "${UNFREEZE_PROJECTOR}" == "1" ]]; then ARGS+=(--unfreeze-projector); else ARGS+=(--freeze-projector); fi
if [[ "${UNFREEZE_DECODER}" == "1" ]]; then ARGS+=(--unfreeze-decoder); fi
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then ARGS+=(--gradient-checkpointing); fi
if [[ "${RUN_FINAL_EVAL:-1}" == "1" ]]; then ARGS+=(--run-final-eval); fi
if [[ "${EVAL_EARLY_STOPPING:-0}" == "1" ]]; then ARGS+=(--eval-early-stopping); fi
if [[ "${WANDB:-1}" == "0" ]]; then ARGS+=(--no-wandb); fi
if [[ -n "${WANDB_RUN_NAME:-}" ]]; then ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}"); fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then ARGS+=(--wandb-entity "${WANDB_ENTITY}"); fi
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS+=(--resume-from "${RESUME_FROM}"); fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then ARGS+=(--no-resume); fi
if [[ -n "${DATASET_NAMES:-}" ]]; then
  # shellcheck disable=SC2206
  DS_ARR=(${DATASET_NAMES})
  ARGS+=(--dataset-names "${DS_ARR[@]}")
fi
if [[ -n "${EVAL_DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  EDS_ARR=(${EVAL_DATASETS})
  ARGS+=(--eval-datasets "${EDS_ARR[@]}")
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching DDP qwen3_slt training on $NUM_GPUS GPUs"
  LOG_DIR="${TORCHRUN_LOG_DIR:-${OUTPUT_DIR}/torchrun_logs}"
  mkdir -p "$LOG_DIR"
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    --log-dir "$LOG_DIR" --redirects 3 --tee 3 \
    "${ARGS[@]}"
else
  echo "Launching single-process qwen3_slt training"
  exec python "${ARGS[@]}"
fi
