#!/usr/bin/env bash
# Train only. For prepare + train + eval, use run_v1_pipeline_full.sh (n6 main entry).
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export PYTHONPATH=.

CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/train.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  ${RESUME:+--resume "$RESUME"}
