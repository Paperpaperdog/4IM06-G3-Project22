#!/usr/bin/env bash
set -e

export PYTHONPATH=.

RAISE_CSV="${RAISE_CSV:-../data/raise_raw/RAISE_1k.csv}"

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
  --output-dir data/processed/v1_fourier_ambiguity \
  --download-dir data/raw/raise_tiff \
  --classes original JPEG_Q80 downsample_x8 downsample_x16 \
  --observed-sizes 128 96 64 48 32 \
  --samples-per-class-per-size 1000 \
  --jpeg-quality 80 \
  --downsample-factors 8 16 \
  --interpolation bicubic \
  --residual tv \
  --tv-weight 0.08 \
  --tv-max-iter 30 \
  --target-spectrum-height 512 \
  --target-spectrum-width-rfft 257 \
  --dc-sigma-bins 3.0 \
  --seed 123 \
  --dtype float16
