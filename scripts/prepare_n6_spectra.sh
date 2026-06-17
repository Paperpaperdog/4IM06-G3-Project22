#!/usr/bin/env bash
# Build the shared 1-channel log-rFFT spectrum cache used by BOTH Mask and CNN.
#
#   SIZE=64 bash scripts/prepare_n6_spectra.sh
#   SIZES="32 64 96 128" bash scripts/prepare_n6_spectra.sh
#   LIMIT_IMAGES=4 SAMPLES_PER_CLASS_PER_SIZE=8 SIZE=64 bash scripts/prepare_n6_spectra.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASK_ROOT="$REPO_ROOT/spectral-mask-resampling"
SIZES="${SIZES:-${SIZE:-64}}"

for size in $SIZES; do
  cfg="$MASK_ROOT/configs/size_sweep/n6_mask_size${size}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "SKIP: missing $cfg" >&2
    continue
  fi
  cache="$REPO_ROOT/data/processed/n6_spectra_size${size}"
  echo "=== prepare shared spectra: size=$size -> $cache ==="
  (
    cd "$MASK_ROOT"
    CONFIG="configs/size_sweep/n6_mask_size${size}.yaml" \
      bash scripts/run_prepare_config.sh
  )
done

echo "Done. Mask and CNN both read: data/processed/n6_spectra_size{N}/"
