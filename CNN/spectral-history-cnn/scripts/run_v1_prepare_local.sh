#!/usr/bin/env bash
# CNN prepare step: use the shared spectrum cache (built once for Mask + CNN).
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export PYTHONPATH=.
export CNN_ROOT="$ROOT"

CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"
repo_root="$(cd "$ROOT/../.." && pwd)"

if [[ ! -f "$ROOT/$CONFIG" ]]; then
  echo "ERROR: config not found: $ROOT/$CONFIG" >&2
  exit 1
fi

# shellcheck disable=SC1091
if [[ -z "${PREP_PY:-}" ]]; then
  source "$ROOT/scripts/resolve_prepare_python.sh"
else
  echo "Using prepare python (inherited): $PREP_PY"
fi
PY="${PREP_PY:-python3}"

PROCESSED_DIR="$("$PY" -c "import yaml; from pathlib import Path; p=yaml.safe_load(open('$CONFIG'))['paths']['processed_dir']; print(Path('$ROOT')/p)")"
PROCESSED_DIR="$(cd "$(dirname "$PROCESSED_DIR")" && pwd)/$(basename "$PROCESSED_DIR")"

if [[ -f "$PROCESSED_DIR/train_spectra.npy" && -f "$PROCESSED_DIR/test_spectra.npy" ]]; then
  echo "Shared spectrum cache OK: $PROCESSED_DIR"
  echo "  (Mask and CNN use the same files — no second preprocess needed)"
  exit 0
fi

echo "Shared cache missing at: $PROCESSED_DIR"
SIZE="$("$PY" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['final_size'])")"
echo "Building via scripts/prepare_n6_spectra.sh (SIZE=$SIZE) ..."
SIZE="$SIZE" exec bash "$repo_root/scripts/prepare_n6_spectra.sh"
