#!/usr/bin/env bash
set -e

export PYTHONPATH=.

RAISE_CSV="${RAISE_CSV:-../RAISE_1k.csv}"

python src/data/split_raise.py \
  --input-csv "${RAISE_CSV}" \
  --url-column TIFF \
  --output-json data/splits/raise_split_seed123.json \
  --train 700 \
  --val 150 \
  --test 150 \
  --seed 123

python src/data/preprocess_spectra.py \
  --split-json data/splits/raise_split_seed123.json \
  --output-dir data/processed/v1_tv_rfft \
  --download-dir data/raw/raise_tiff \
  --patch-size 512 \
  --patches-per-image 10 \
  --jpeg-quality 80 \
  --downsample-factors 2 4 8 16 \
  --interpolation bicubic \
  --residual tv \
  --tv-weight 0.08 \
  --tv-max-iter 30 \
  --dc-sigma-bins 3.0 \
  --seed 123 \
  --dtype float16
