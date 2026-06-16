#!/usr/bin/env bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

python src/evaluate.py \
  --config configs/v1_fourier_ambiguity_mask_clean.yaml \
  --checkpoint outputs/v1_fourier_ambiguity_mask_clean/checkpoints/best.pt \
  --split test

python src/visualize.py \
  --config configs/v1_fourier_ambiguity_mask_clean.yaml \
  --checkpoint outputs/v1_fourier_ambiguity_mask_clean/checkpoints/best.pt
