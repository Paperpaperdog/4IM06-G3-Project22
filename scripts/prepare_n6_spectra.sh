#!/usr/bin/env bash
# Build the shared 1-channel log-rFFT spectrum cache used by BOTH Mask and CNN.
#
#   SIZE=64 bash scripts/prepare_n6_spectra.sh
#   SIZES="32 64 96 128" bash scripts/prepare_n6_spectra.sh
#   PARALLEL_SIZES=1 bash scripts/prepare_n6_spectra.sh   # four sizes at once
#   bash scripts/submit_prepare_n6.sh --detach --parallel  # background + parallel
#
# For cluster CPU jobs see: scripts/submit_prepare_n6.sh --cluster
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
SIZES="${SIZES:-${SIZE:-64}}"
PREP_WORKERS="${PREP_WORKERS:-0}"
PARALLEL_SIZES="${PARALLEL_SIZES:-0}"

if [[ "$PARALLEL_SIZES" == "1" ]]; then
  exec bash "$REPO_ROOT/scripts/submit_prepare_n6.sh" --parallel
fi

for size in $SIZES; do
  SIZE="$size" PREP_WORKERS="$PREP_WORKERS" \
    bash "$REPO_ROOT/scripts/prepare_n6_worker.sh"
done

echo "Done. Mask and CNN both read: data/processed/n6_spectra_size{N}/"
