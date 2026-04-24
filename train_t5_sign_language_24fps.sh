#!/usr/bin/env bash
set -euo pipefail

cd /home/slimelab/Projects/slt/src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-sign-t5-slt}"
export WANDB_NAME="${WANDB_NAME:-t5_sign_language_24fps}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data4/sign_t5_runs/t5_sign_language_24fps}"
DATA_ROOT="${DATA_ROOT:-/mnt/data2/sign_language_24fps}"
METADATA_ROOT="${METADATA_ROOT:-/home/slimelab/Projects/slt/islr/webdataset_224}"

exec torchrun --standalone --nproc_per_node=4 -m t5_slt.train \
  --dataset-format webdataset_tar \
  --data-root "${DATA_ROOT}" \
  --metadata-root "${METADATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name-or-path google/t5-v1_1-small \
  --feature-dim 1104 \
  --prompt-template "translate {sign_language} to {language}" \
  --num-train-epochs 30 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --train-batch-size 64 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 1 \
  --max-source-length 256 \
  --max-target-length 48 \
  --generation-max-length 48 \
  --num-beams 4 \
  --logging-steps 20 \
  --dataloader-num-workers 8 \
  --report-to wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_NAME}" \
  --bf16
