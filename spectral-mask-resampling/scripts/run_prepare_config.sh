#!/usr/bin/env bash
# Preprocess the spectra cache for ONE mask config (reads classes / observed
# sizes / data_dir straight from the YAML), so per-size sweep configs each get
# their own cache. CPU-only step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

CONFIG="${CONFIG:?Set CONFIG=configs/size_sweep/n6_mask_size64.yaml}"
repo_root="$(cd "$ROOT/.." && pwd)"
RAISE_TIFF_DIR="${RAISE_TIFF_DIR:-$ROOT/data/raw/raise_tiff}"
if [[ ! -d "$RAISE_TIFF_DIR" && -d "${repo_root/-integration/}/spectral-mask-resampling/data/raw/raise_tiff" ]]; then
  RAISE_TIFF_DIR="${repo_root/-integration/}/spectral-mask-resampling/data/raw/raise_tiff"
fi
export RAISE_TIFF_DIR
# Canonical train/val/test lists for Mask **and** CNN (mask split_raise.py, seed=123).
SPLIT_JSON="${SPLIT_JSON:-data/splits/raise_split_seed123.json}"
SAMPLES_PER_CLASS_PER_SIZE="${SAMPLES_PER_CLASS_PER_SIZE:-1000}"
# 0 = use all CPU cores for the (size, class) block parallelism.
PREP_WORKERS="${PREP_WORKERS:-0}"

# Prepare needs PIL/numpy/torch — use project venv, not bare python3.
if [[ -z "${PYTHON:-}" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_python.sh" cpu
else
  echo "Using prepare python (inherited): $PYTHON"
fi
PY="${PYTHON:-python3}"

if [[ ! -d "${RAISE_TIFF_DIR}" ]]; then
  echo "Missing TIFF directory: ${RAISE_TIFF_DIR}" >&2
  echo "Run scripts/download_raise_tiff.sh or point RAISE_TIFF_DIR at the cache." >&2
  exit 1
fi

# Extract classes, observed sizes and the data_dir from the config (one field per line;
# a single `read` line would split class names incorrectly).
mapfile -t _cfg_lines < <("$PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["data_dir"])
print(" ".join(cfg["class_names"]))
print(" ".join(str(s) for s in cfg["observed_sizes"]))
PY
)
DATA_DIR="${_cfg_lines[0]}"
CLASSES="${_cfg_lines[1]}"
OBSERVED_SIZES="${_cfg_lines[2]}"

echo "CONFIG=$CONFIG"
echo "DATA_DIR=$DATA_DIR"
echo "CLASSES=$CLASSES"
echo "OBSERVED_SIZES=$OBSERVED_SIZES (native rFFT spectrum)"

if [[ ! -f "${SPLIT_JSON}" ]]; then
  "$PY" src/data/split_raise.py \
    --input-dir "${RAISE_TIFF_DIR}" \
    --output-json "${SPLIT_JSON}" \
    --train 700 --val 150 --test 150 --seed 123
fi

# shellcheck disable=SC2086
"$PY" src/data/preprocess_spectra.py \
  --split-json "${SPLIT_JSON}" \
  --input-dir "${RAISE_TIFF_DIR}" \
  --output-dir "${DATA_DIR}" \
  --download-dir "${RAISE_TIFF_DIR}" \
  --classes ${CLASSES} \
  --observed-sizes ${OBSERVED_SIZES} \
  --samples-per-class-per-size "${SAMPLES_PER_CLASS_PER_SIZE}" \
  --jpeg-quality 80 \
  --interpolation bicubic \
  --residual tv \
  --tv-weight 0.08 \
  --tv-max-iter 30 \
  --dc-sigma-bins 3.0 \
  --seed 123 \
  --dtype float16 \
  --workers "${PREP_WORKERS}" \
  ${LIMIT_IMAGES:+--limit-images "$LIMIT_IMAGES"}
