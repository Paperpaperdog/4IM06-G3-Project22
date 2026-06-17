#!/usr/bin/env bash
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export PYTHONPATH=.
export CNN_ROOT="$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/resolve_prepare_python.sh"

CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
RAISE_DIR="${RAISE_DIR:-../../spectral-mask-resampling/data/raw/raise_tiff}"
SPLIT_JSON="${SPLIT_JSON:-data/splits/raise_split_seed123_local.json}"

if [[ ! -f "$ROOT/$CONFIG" ]]; then
  echo "ERROR: config not found: $ROOT/$CONFIG" >&2
  echo "  CNN_ROOT=$ROOT" >&2
  exit 1
fi

RAISE_DIR="$(cd "$RAISE_DIR" && pwd)"

if [[ ! -d "$RAISE_DIR" ]]; then
  echo "ERROR: RAISE TIFF directory not found: $RAISE_DIR" >&2
  exit 1
fi

tiff_count="$(find "$RAISE_DIR" -maxdepth 1 -iname '*.tif' | wc -l)"
echo "Using local RAISE cache: $RAISE_DIR ($tiff_count TIFF files)"

mkdir -p data/splits

"$PREP_PY" src/data/split_raise.py \
  --input-dir "$RAISE_DIR" \
  --output-json "$SPLIT_JSON" \
  --train 700 \
  --val 150 \
  --test 150 \
  --seed 123

"$PREP_PY" src/data/preprocess_spectra.py \
  --config "$CONFIG" \
  --raise-dir "$RAISE_DIR" \
  --image-cache-dir "$RAISE_DIR" \
  --split-json "$SPLIT_JSON" \
  --workers "${WORKERS:-32}" \
  ${LIMIT_SAMPLES:+--limit-samples "$LIMIT_SAMPLES"}
