#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/evaluate.py \
  --config configs/v1_final64_poscnn.yaml \
  --checkpoint outputs/v1_final64_poscnn/checkpoints/best.pt \
  --split test

python src/visualize.py \
  --config configs/v1_final64_poscnn.yaml \
  --checkpoint outputs/v1_final64_poscnn/checkpoints/best.pt \
  --split test
