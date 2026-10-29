#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash train.sh GPU_IDS [training arguments...]"
    exit 1
fi

GPU_IDS="$1"
shift

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    python train_val.py "$@" --seed 444