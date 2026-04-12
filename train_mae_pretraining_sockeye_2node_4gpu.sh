#!/usr/bin/env bash
set -euo pipefail

cd /scratch/st-jzhu71-1/jzhu71/slt

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NNODES="${SLURM_NNODES:-2}"
export NPROC_PER_NODE=4
export NODE_RANK="${SLURM_NODEID:?SLURM_NODEID is required}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec apptainer exec --nv --cleanenv -C \
  --bind /scratch/st-jzhu71-1/jzhu71:/scratch/st-jzhu71-1/jzhu71 \
  --bind /arc:/arc \
  pytorch-sign.sif \
  bash -c '
    cd /scratch/st-jzhu71-1/jzhu71/slt/src
    export CUDA_VISIBLE_DEVICES='"${CUDA_VISIBLE_DEVICES}"'
    export MASTER_ADDR='"${MASTER_ADDR}"'
    export MASTER_PORT='"${MASTER_PORT}"'
    export NODE_RANK='"${NODE_RANK}"'
    export NNODES='"${NNODES}"'
    exec torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node=4 \
      --node_rank='"${NODE_RANK}"' \
      --master_addr='"${MASTER_ADDR}"' \
      --master_port='"${MASTER_PORT}"' \
      -m mae_pretraining.run_pretraining \
      data.base_data_dir=/arc/project/st-jzhu71-1/sign_language \
      data.dataset_names=all_train_plain_v3 \
      common.output_dir=mae_pretraining_runs/run_001_2nodes \
      common.log_dir=mae_pretraining_runs/run_001_2nodes/logs
  '
