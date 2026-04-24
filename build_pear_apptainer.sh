#!/usr/bin/env bash
set -e

# Simplified build script
SIF_FILE="pear_v1.sif"
DEF_FILE="pear_apptainer.def"

echo "Building $SIF_FILE from $DEF_FILE..."
apptainer build --force --fakeroot "$SIF_FILE" "$DEF_FILE"

echo "Build complete."
