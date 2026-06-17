#!/usr/bin/env bash
# On-node entrypoint for the mask pipeline on an Ascend NPU compute node.
# Analogous to CNN/spectral-history-cnn/scripts/run_v1_gpu.sh. A cluster vc
# wrapper (e.g. vc_mask_u6.sh) should set CONFIG and exec this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$(cd "$ROOT/.." && pwd)}"
export REPO_ROOT
export CNN_ROOT="${CNN_ROOT:-$REPO_ROOT/CNN/spectral-history-cnn}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT/-integration/}/spectral-mask-resampling/.venv}"
legacy_repo="${REPO_ROOT/-integration/}"
RAISE_TIFF_DIR="${RAISE_TIFF_DIR:-$ROOT/data/raw/raise_tiff}"
if [[ ! -d "$RAISE_TIFF_DIR" && -d "$legacy_repo/spectral-mask-resampling/data/raw/raise_tiff" ]]; then
  export RAISE_TIFF_DIR="$legacy_repo/spectral-mask-resampling/data/raw/raise_tiff"
else
  export RAISE_TIFF_DIR
fi

export CONFIG="${CONFIG:-configs/u6_mask_combined.yaml}"
export SKIP_PREPARE="${SKIP_PREPARE:-0}"
export EVAL_ONLY="${EVAL_ONLY:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
if grep -qE '^[[:space:]]*device:[[:space:]]*npu' "$ROOT/$CONFIG" 2>/dev/null; then
  export PYTHONNOUSERSITE=1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vc_mask_${TS}.log"

echo "[$(date '+%F %T')] mask vc_worker start CONFIG=$CONFIG" | tee "$LOG_FILE"
echo "REPO_ROOT=$REPO_ROOT CNN_ROOT=$CNN_ROOT VENV_DIR=$VENV_DIR RAISE_TIFF_DIR=$RAISE_TIFF_DIR SKIP_PREPARE=$SKIP_PREPARE EVAL_ONLY=$EVAL_ONLY" | tee -a "$LOG_FILE"
bash "$ROOT/scripts/run_pipeline_config.sh" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] mask vc_worker done" | tee -a "$LOG_FILE"
