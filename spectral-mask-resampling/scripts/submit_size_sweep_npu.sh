#!/usr/bin/env bash
# Submit ONE NPU job per observed size for the unified 6-class mask sweep.
#
# Like the CNN sweep, this defers to a cluster-side vc wrapper (default name
# vc_mask_u6.sh, sitting next to vc_cnn_spectral_v1.sh). That wrapper should
# `exec` spectral-mask-resampling/scripts/vc_worker.sh with the env it receives.
#
#   bash scripts/submit_size_sweep_npu.sh
#   SIZES="64 128" bash scripts/submit_size_sweep_npu.sh
#   VC_WRAPPER=vc_mask_u6.sh CODES=/path/to/Codes bash scripts/submit_size_sweep_npu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"
VC_WRAPPER="${VC_WRAPPER:-vc_mask_u6.sh}"
SIZES="${SIZES:-32 64 96 128}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

if [[ ! -d "$CODES" ]]; then
  echo "ERROR: cluster Codes dir not found: $CODES" >&2
  echo "Set CODES=... to the directory holding the vc wrappers." >&2
  exit 1
fi
if [[ ! -f "$CODES/$VC_WRAPPER" ]]; then
  echo "ERROR: cluster vc wrapper not found: $CODES/$VC_WRAPPER" >&2
  echo "Create it next to vc_cnn_spectral_v1.sh; it only needs to exec:" >&2
  echo "  bash <repo>/spectral-mask-resampling/scripts/vc_worker.sh" >&2
  exit 1
fi

for size in $SIZES; do
  cfg="configs/size_sweep/u6_mask_size${size}.yaml"
  if [[ ! -f "$ROOT/$cfg" ]]; then
    echo "SKIP: missing config $ROOT/$cfg" >&2
    continue
  fi
  echo "=== submit mask size=$size config=$cfg ==="
  cd "$CODES"
  CONFIG="$cfg" \
  SKIP_PREPARE="$SKIP_PREPARE" \
  JOB="mask_u6_size${size}" \
  bash "$VC_WRAPPER"
done

echo "Submitted mask size-sweep jobs for sizes: $SIZES"
echo "Monitor with: vc list"
