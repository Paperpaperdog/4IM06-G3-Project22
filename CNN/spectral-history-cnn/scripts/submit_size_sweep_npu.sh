#!/usr/bin/env bash
# Submit ONE NPU job per observed size for the unified 6-class CNN sweep, using
# the same vc wrapper as the rest of the project (vc_cnn_spectral_v1.sh on the
# cluster). Mirrors scripts/submit_npu_train.sh / resubmit_parallel_full.sh.
#
#   bash scripts/submit_size_sweep_npu.sh
#   SIZES="64 128" EPOCHS=50 bash scripts/submit_size_sweep_npu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"
SIZES="${SIZES:-32 64 96 128}"
EPOCHS="${EPOCHS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

if [[ ! -d "$CODES" ]]; then
  echo "ERROR: cluster Codes dir not found: $CODES" >&2
  echo "Set CODES=... to the directory holding vc_cnn_spectral_v1.sh." >&2
  exit 1
fi

for size in $SIZES; do
  cfg="configs/size_sweep/u6_poscnn_size${size}.yaml"
  if [[ ! -f "$ROOT/$cfg" ]]; then
    echo "SKIP: missing config $ROOT/$cfg" >&2
    continue
  fi
  echo "=== submit CNN size=$size config=$cfg ==="
  cd "$CODES"
  CONFIG="$cfg" \
  EPOCHS="$EPOCHS" \
  SKIP_PREPARE="$SKIP_PREPARE" \
  JOB="cnn_u6_size${size}" \
  bash vc_cnn_spectral_v1.sh
done

echo "Submitted CNN size-sweep jobs for sizes: $SIZES"
echo "Monitor with: vc list"
