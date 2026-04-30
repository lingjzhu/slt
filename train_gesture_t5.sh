#!/usr/bin/env bash
# Three-stage training of T5 over a pretrained gesture encoder.
#   stage 1: train MLP connector only        (freeze gesture encoder + T5)
#   stage 2: train MLP connector + encoder   (freeze T5)
#   stage 3: train all modules
#
# Each stage launches an independent torchrun job and loads
# `pytorch_model.bin` from the previous stage's output directory.
#
# Override STAGES="1" or "2 3" etc. to run a subset of stages.
set -euo pipefail

cd /home/slimelab/Projects/slt/src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_PROJECT="${WANDB_PROJECT:-sign-t5-slt}"

if command -v nproc >/dev/null 2>&1; then
  CPU_COUNT="$(nproc)"
else
  CPU_COUNT=8
fi
NUM_GPUS="${NUM_GPUS:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPU_COUNT > NUM_GPUS ? CPU_COUNT / NUM_GPUS : 1 ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then OMP_NUM_THREADS=1; fi
export OMP_NUM_THREADS
MASTER_PORT="${MASTER_PORT:-29514}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data4/sign_t5_runs/t5_gesture_slt}"
DATA_ROOT="${DATA_ROOT:-/mnt/data2/sign_gestures}"
GESTURE_CHECKPOINT="${GESTURE_CHECKPOINT:-/mnt/data4/outputs/gesture_pretraining_base/step-0040000.pt}"
MODEL_NAME="${MODEL_NAME:-google/byt5-small}"
# Empty DATASETS → all available gesture datasets are used.
DATASETS="${DATASETS:-}"

# Per-stage hyperparams. Override via env to tune.
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-5e-4}"
STAGE3_LR="${STAGE3_LR:-5e-4}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
# byt5 emits byte-level tokens; English ≈1.5×, Chinese (UTF-8) ≈3× sentencepiece length.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-64}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-256}"
GEN_MAX_LENGTH="${GEN_MAX_LENGTH:-128}"
NUM_BEAMS="${NUM_BEAMS:-4}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-4}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.15}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.8}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
FULL_BF16="${FULL_BF16:-1}"
# Default eval is the explicit final val/test pass, which prints progress bars.
# Set EVAL_STRATEGY=epoch or steps to also run HF Trainer eval during training.
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
STREAMING_SAMPLES_PER_EPOCH="${STREAMING_SAMPLES_PER_EPOCH:-200000}"
STREAMING_SHUFFLE_BUFFER="${STREAMING_SHUFFLE_BUFFER:-2000}"
# Shared fallback. Prefer STAGE{1,2,3}_MAX_STEPS for streaming training.
MAX_STEPS="${MAX_STEPS:-}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-${MAX_STEPS:-3000}}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-${MAX_STEPS:-15000}}"
STAGE3_MAX_STEPS="${STAGE3_MAX_STEPS:-${MAX_STEPS:-100000}}"

STAGES="${STAGES:-1 2 3}"

run_stage () {
  local stage="$1"
  local lr="$2"
  local max_steps="$3"
  local stage_out="${OUTPUT_ROOT}/stage${stage}"
  local wandb_name="${WANDB_NAME:-t5_gesture_slt}_stage${stage}"
  local extra_args=()

  if [[ -n "$EVAL_STEPS" ]]; then
    extra_args+=(--eval-steps "$EVAL_STEPS")
  fi
  if [[ "$RUN_FINAL_EVAL" == "1" ]]; then
    extra_args+=(--run-final-eval)
  fi
  if [[ "$FULL_BF16" == "1" ]]; then
    extra_args+=(--full-bf16)
  fi

  if [[ "$stage" -gt 1 ]]; then
    # By default each stage > 1 picks up the latest checkpoint from the
    # previous stage's output directory. Override via STAGE{2,3}_LOAD_FROM.
    local override_var="STAGE${stage}_LOAD_FROM"
    local override_val="${!override_var:-}"
    local prev_dir="${override_val:-${OUTPUT_ROOT}/stage$(( stage - 1 ))}"
    if [[ ! -e "${prev_dir}" ]]; then
      echo "ERROR: stage ${stage} expected to load from ${prev_dir} but it does not exist." >&2
      echo "  Run the previous stage first or set ${override_var}=<path>." >&2
      exit 1
    fi
    extra_args+=(--load-from "${prev_dir}")
    echo "[stage ${stage}] loading from ${prev_dir}"
  fi

  echo ""
  echo "================================================================"
  echo " STAGE ${stage}  →  ${stage_out}  (max_steps=${max_steps}, lr=${lr})"
  echo "================================================================"

  mkdir -p "${stage_out}"

  torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m t5_slt.train \
    --dataset-format gesture_translation \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --output-dir "${stage_out}" \
    --model-name-or-path "${MODEL_NAME}" \
    --use-gesture-encoder \
    --gesture-checkpoint "${GESTURE_CHECKPOINT}" \
    --gesture-in-dim 1104 \
    --gesture-hidden-size 768 \
    --gesture-num-hidden-layers 22 \
    --gesture-num-attention-heads 12 \
    --gesture-intermediate-size 1152 \
    --gesture-max-position-embeddings 1024 \
    --gesture-global-attn-every-n-layers 3 \
    --gesture-local-attention 128 \
    --projection-hidden-dim 1024 \
    --projection-dropout 0.1 \
    --training-stage "${stage}" \
    --prompt-template "translate {sign_language} to {language}" \
    --learning-rate "${lr}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --max-source-length "${MAX_SOURCE_LENGTH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --max-target-length "${MAX_TARGET_LENGTH}" \
    --generation-max-length "${GEN_MAX_LENGTH}" \
    --num-beams "${NUM_BEAMS}" \
    --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}" \
    --repetition-penalty "${REPETITION_PENALTY}" \
    --length-penalty "${LENGTH_PENALTY}" \
    --early-stopping \
    --logging-steps "${LOGGING_STEPS}" \
    --max-steps "${max_steps}" \
    --dataloader-num-workers "${NUM_WORKERS}" \
    --eval-strategy "${EVAL_STRATEGY}" \
    --streaming-samples-per-epoch "${STREAMING_SAMPLES_PER_EPOCH}" \
    --streaming-shuffle-buffer "${STREAMING_SHUFFLE_BUFFER}" \
    --report-to "${REPORT_TO:-wandb}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${wandb_name}" \
    --gradient-checkpointing \
    --bf16 \
    "${extra_args[@]}"
}

for s in $STAGES; do
  case "$s" in
    1) run_stage 1 "${STAGE1_LR}" "${STAGE1_MAX_STEPS}" ;;
    2) run_stage 2 "${STAGE2_LR}" "${STAGE2_MAX_STEPS}" ;;
    3) run_stage 3 "${STAGE3_LR}" "${STAGE3_MAX_STEPS}" ;;
    *) echo "Unknown stage: $s" >&2; exit 1 ;;
  esac
done
