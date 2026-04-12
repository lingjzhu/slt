#!/usr/bin/env bash
set -euo pipefail

cd /home/slimelab/Projects/slt/src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_PROJECT="${WANDB_PROJECT:-sign-t5-slt}"
export WANDB_NAME="${WANDB_NAME:-t5_signhiera_run001}"
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data4/sign_t5_runs/run_001}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/mnt/data4/all_train_plain_v3_mae_features_run_001_8fps/manifests/train_feature_detailed.tsv}"
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/data4/all_val_plain_v3_mae_features_run_001_8fps/manifests/val_feature_detailed.tsv}"
TEST_MANIFEST="${TEST_MANIFEST:-/mnt/data4/all_test_plain_v3_mae_features_run_001_8fps/manifests/test_feature_detailed.tsv}"

exec torchrun --standalone --nproc_per_node=4 -m t5_slt.train \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name-or-path google/t5-v1_1-base \
  --feature-dim 768 \
  --prompt-template "translate to {language}" \
  --attn-implementation sdpa \
  --num-train-epochs 10 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --max-source-length 512 \
  --max-target-length 128 \
  --generation-max-length 128 \
  --num-beams 4 \
  --logging-steps 25 \
  --dataloader-num-workers 8 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_NAME}"
