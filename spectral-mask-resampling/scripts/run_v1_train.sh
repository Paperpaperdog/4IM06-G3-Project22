#!/usr/bin/env bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

python src/train.py \
  --config configs/v1_fourier_ambiguity_mask_clean.yaml
