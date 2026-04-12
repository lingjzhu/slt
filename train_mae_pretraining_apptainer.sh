#!/usr/bin/env bash
set -euo pipefail

cd /home/slimelab/Projects/slt
export CUDA_VISIBLE_DEVICES=0,1,2,3

exec apptainer exec --nv \
  --bind /home/slimelab/Projects:/home/slimelab/Projects \
  --bind /mnt/data4:/mnt/data4 \
  /home/slimelab/Projects/slt/pytorch-sign.sif \
  bash -lc '
    cd /home/slimelab/Projects/slt/src
    export CUDA_VISIBLE_DEVICES='"${CUDA_VISIBLE_DEVICES}"'
    exec torchrun --standalone --nproc_per_node=4 -m mae_pretraining.run_pretraining \
      data.base_data_dir=/mnt/data4 \
      data.dataset_names=all_train_plain_v3 \
      common.output_dir=/mnt/data4/mae_pretraining_runs/run_001 \
      common.log_dir=/mnt/data4/mae_pretraining_runs/run_001/logs
  '
