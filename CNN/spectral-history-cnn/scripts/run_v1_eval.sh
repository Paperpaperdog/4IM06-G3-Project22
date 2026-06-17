#!/usr/bin/env bash
# Eval + visualize only. Main n6 entry: run_v1_pipeline_full.sh with CONFIG=...
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export PYTHONPATH=.

CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SPLIT="${SPLIT:-test}"

OUTPUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['paths']['output_dir'])")"
CKPT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/best.pt}"

python src/evaluate.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CKPT" \
  --split "$SPLIT"

python src/visualize.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CKPT" \
  --split "$SPLIT"
