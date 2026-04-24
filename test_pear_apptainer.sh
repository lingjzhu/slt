#!/usr/bin/env bash
set -e

SIF_FILE="pear_v1.sif"

if [ ! -f "$SIF_FILE" ]; then
    echo "Error: $SIF_FILE not found. Build it first."
    exit 1
fi

echo "Testing Apptainer image: $SIF_FILE"

# Run a simple python command to check imports
apptainer exec "$SIF_FILE" python -c "
import torch
import torchvision
import cv2
import av
import pytorch3d
import smplx
import webdataset
import lightning
print('--- Test Results ---')
print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('All critical imports successful!')
"
