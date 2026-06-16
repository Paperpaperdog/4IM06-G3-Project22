#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/train.py \
  --config configs/v1_final64_poscnn.yaml
