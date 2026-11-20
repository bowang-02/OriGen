#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0,1}
TRAIN_DATA=${TRAIN_DATA:-dataset/train.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PYTHON_BIN" -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --gpu_ids "$GPU_IDS" --mixed_precision bf16 \
  -m train_OriGen.train \
  --train-data "$TRAIN_DATA" \
  --fusion-model-path "$OUTPUT_ROOT/train_fusion_only/final" \
  --output-dir "$OUTPUT_ROOT/OriGen" "$@"
