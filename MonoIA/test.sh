#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash test.sh GPU_IDS --config CONFIG --checkpoint CHECKPOINT [other arguments]"
    exit 1
fi

GPU_IDS="$1"
shift

if [ "$#" -eq 0 ]; then
    echo "ERROR: Evaluation arguments are required."
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    python train_val.py --evaluate_only "$@"
