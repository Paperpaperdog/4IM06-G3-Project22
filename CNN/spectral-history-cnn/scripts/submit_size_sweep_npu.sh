#!/usr/bin/env bash
# Submit ONE NPU job per observed size for the unified 6-class CNN sweep, using
# the same vc wrapper as the rest of the project (vc_cnn_spectral_v1.sh on the
# cluster). Mirrors scripts/submit_npu_train.sh / resubmit_parallel_full.sh.
#
#   bash scripts/submit_size_sweep_npu.sh
#   SWEEP_TAG=u7 bash scripts/submit_size_sweep_npu.sh   # 7-class (+ upsample_x8)
#   SWEEP_TAG=n6 bash scripts/submit_size_sweep_npu.sh   # native-spectrum 6-class (ds8/ds16/up4/up8)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CNN_ROOT="${CNN_ROOT:-$(cd "$ROOT" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$ROOT/../.." && pwd)}"
# RAISE TIFF may live in the non-integration checkout; pass explicitly if needed.
RAISE_DIR="${RAISE_DIR:-$REPO_ROOT/spectral-mask-resampling/data/raw/raise_tiff}"
if [[ ! -d "$RAISE_DIR" && -d "${REPO_ROOT/-integration/}/spectral-mask-resampling/data/raw/raise_tiff" ]]; then
  RAISE_DIR="${REPO_ROOT/-integration/}/spectral-mask-resampling/data/raw/raise_tiff"
fi
# CPU prepare needs scikit-image; reuse the mask project venv on this cluster.
VENV_DIR="${VENV_DIR:-${REPO_ROOT/-integration/}/spectral-mask-resampling/.venv}"
CODES="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"
SIZES="${SIZES:-32 64 96 128}"
EPOCHS="${EPOCHS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SWEEP_TAG="${SWEEP_TAG:-u6}"

if [[ "$SWEEP_TAG" != "u6" && "$SWEEP_TAG" != "u7" && "$SWEEP_TAG" != "n6" ]]; then
  echo "ERROR: SWEEP_TAG must be u6, u7 or n6 (got $SWEEP_TAG)" >&2
  exit 1
fi

if [[ ! -d "$CODES" ]]; then
  echo "ERROR: cluster Codes dir not found: $CODES" >&2
  echo "Set CODES=... to the directory holding vc_cnn_spectral_v1.sh." >&2
  exit 1
fi

for size in $SIZES; do
  cfg="configs/size_sweep/${SWEEP_TAG}_poscnn_size${size}.yaml"
  if [[ ! -f "$ROOT/$cfg" ]]; then
    echo "SKIP: missing config $ROOT/$cfg" >&2
    continue
  fi
  echo "=== submit CNN size=$size config=$cfg CNN_ROOT=$CNN_ROOT ==="
  cd "$CODES"
  CNN_ROOT="$CNN_ROOT" \
  RAISE_DIR="$RAISE_DIR" \
  VENV_DIR="$VENV_DIR" \
  CONFIG="$cfg" \
  EPOCHS="$EPOCHS" \
  SKIP_PREPARE="$SKIP_PREPARE" \
  JOB="${SWEEP_TAG}_cnn_size${size}" \
  bash vc_cnn_spectral_v1.sh
done

echo "Submitted CNN size-sweep jobs (SWEEP_TAG=$SWEEP_TAG) for sizes: $SIZES"
echo "Monitor with: vc list"
