#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

RAISE_CSV="${RAISE_CSV:-../../data/raise_raw/RAISE_1k.csv}"

python src/data/split_raise.py \
  --csv "$RAISE_CSV" \
  --id-column File \
  --url-column TIFF \
  --output-json data/splits/raise_split_seed123.json \
  --train 700 \
  --val 150 \
  --test 150 \
  --seed 123

python src/data/preprocess_spectra.py \
  --config configs/v1_final64_poscnn.yaml \
  --split-json data/splits/raise_split_seed123.json \
  --image-cache-dir data/raw/raise_tiff
