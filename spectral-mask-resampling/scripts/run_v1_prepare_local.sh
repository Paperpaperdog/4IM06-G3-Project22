#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

RAISE_TIFF_DIR="${RAISE_TIFF_DIR:-data/raw/raise_tiff}"
SPLIT_JSON="${SPLIT_JSON:-data/splits/raise_split_seed123.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/v1_fourier_ambiguity}"

if [[ ! -d "${RAISE_TIFF_DIR}" ]]; then
  echo "Missing TIFF directory: ${RAISE_TIFF_DIR}" >&2
  echo "Download on a machine with internet access, then sync to the cluster:" >&2
  echo "  bash scripts/download_raise_tiff.sh" >&2
  exit 1
fi

num_tiff="$(find "${RAISE_TIFF_DIR}" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l | tr -d ' ')"
if [[ "${num_tiff}" -lt 1000 ]]; then
  echo "Warning: found only ${num_tiff} TIFF files in ${RAISE_TIFF_DIR}; need at least 1000." >&2
fi

python src/data/split_raise.py \
  --input-dir "${RAISE_TIFF_DIR}" \
  --output-json "${SPLIT_JSON}" \
  --train 700 \
  --val 150 \
  --test 150 \
  --seed 123

python src/data/preprocess_spectra.py \
  --split-json "${SPLIT_JSON}" \
  --input-dir "${RAISE_TIFF_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --download-dir "${RAISE_TIFF_DIR}" \
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
