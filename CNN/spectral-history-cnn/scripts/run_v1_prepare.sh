#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

RAISE_DIR="${RAISE_DIR:-/path/to/RAISE-1k}"

python src/data/split_raise.py \
  --input-dir "$RAISE_DIR" \
  --output-json data/splits/raise_split_seed123.json \
  --train 700 \
  --val 150 \
  --test 150 \
  --seed 123

python src/data/preprocess_spectra.py \
  --config configs/v1_final64_poscnn.yaml \
  --raise-dir "$RAISE_DIR" \
  --split-json data/splits/raise_split_seed123.json
