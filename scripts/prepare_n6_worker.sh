#!/usr/bin/env bash
# On-node worker: build shared spectrum cache for ONE observed size.
# Called by prepare_n6_spectra.sh, submit_prepare_n6.sh, or vc_prepare_n6.sh.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MASK_ROOT="$REPO_ROOT/spectral-mask-resampling"
SIZE="${SIZE:?Set SIZE=32|64|96|128}"
PREP_WORKERS="${PREP_WORKERS:-0}"

cfg="$MASK_ROOT/configs/size_sweep/n6_mask_size${SIZE}.yaml"
if [[ ! -f "$cfg" ]]; then
  echo "ERROR: missing config $cfg" >&2
  exit 1
fi

cache="$REPO_ROOT/data/processed/n6_spectra_size${SIZE}"
if [[ -f "$cache/train_spectra.npy" && -f "$cache/test_spectra.npy" && "${FORCE_PREPARE:-0}" != "1" ]]; then
  echo "SKIP: cache already exists at $cache (set FORCE_PREPARE=1 to rebuild)"
  exit 0
fi

echo "[$(date '+%F %T')] prepare size=$SIZE workers=$PREP_WORKERS -> $cache"
(
  cd "$MASK_ROOT"
  CONFIG="configs/size_sweep/n6_mask_size${SIZE}.yaml" \
    PREP_WORKERS="$PREP_WORKERS" \
    LIMIT_IMAGES="${LIMIT_IMAGES:-}" \
    SAMPLES_PER_CLASS_PER_SIZE="${SAMPLES_PER_CLASS_PER_SIZE:-}" \
    bash scripts/run_prepare_config.sh
)
echo "[$(date '+%F %T')] done size=$SIZE"
