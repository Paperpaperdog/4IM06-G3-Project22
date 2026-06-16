#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

RAISE_CSV="${RAISE_CSV:-../RAISE_1k.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/raw/raise_tiff}"

if [[ -n "${LIMIT:-}" ]]; then
  python src/data/download_raise_tiff.py \
    --input-csv "${RAISE_CSV}" \
    --url-column TIFF \
    --output-dir "${OUTPUT_DIR}" \
    --limit "${LIMIT}"
else
  python src/data/download_raise_tiff.py \
    --input-csv "${RAISE_CSV}" \
    --url-column TIFF \
    --output-dir "${OUTPUT_DIR}"
fi

echo "TIFF cache ready at ${OUTPUT_DIR}"
echo "File count: $(find "${OUTPUT_DIR}" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l)"
