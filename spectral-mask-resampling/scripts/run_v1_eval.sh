#!/usr/bin/env bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

python src/evaluate.py \
  --config configs/v1_tv_rfft_mask.yaml \
  --checkpoint outputs/v1_tv_rfft_mask/checkpoints/best.pt \
  --split test

python src/visualize.py \
  --config configs/v1_tv_rfft_mask.yaml \
  --checkpoint outputs/v1_tv_rfft_mask/checkpoints/best.pt
