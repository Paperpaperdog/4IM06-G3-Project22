#!/usr/bin/env bash
# Worker for vc submit on Ascend NPU nodes (pdgpu-sjtu-ai).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# submit_size_sweep_npu.sh passes CNN_ROOT; fallback to this script's checkout.
CNN_ROOT="${CNN_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"
ROOT="$(cd "$CNN_ROOT" && pwd)"
cd "$ROOT"
export CNN_ROOT="$ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
export RAISE_DIR="${RAISE_DIR:-$ROOT/../../spectral-mask-resampling/data/raw/raise_tiff}"
export DEVICE="${DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Avoid ~/.local CUDA torch shadowing container torch_npu on NPU nodes.
if [[ "$DEVICE" == "npu" ]]; then
  export PYTHONNOUSERSITE=1
fi
export WORKERS="${WORKERS:-18}"
export SKIP_PREPARE="${SKIP_PREPARE:-0}"
export RESUME="${RESUME:-}"
# CPU prepare on NPU nodes needs mask venv (scikit-image); pass through to pipeline.
repo_for_venv="$(cd "$ROOT/../.." && pwd)"
export VENV_DIR="${VENV_DIR:-${repo_for_venv/-integration/}/spectral-mask-resampling/.venv}"

TS="$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vc_npu_${TS}.log"

echo "[$(date '+%F %T')] run_v1_gpu (NPU) start" | tee "$LOG_FILE"
echo "ROOT=$ROOT CONFIG=$CONFIG DEVICE=$DEVICE ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "RAISE_DIR=$RAISE_DIR VENV_DIR=$VENV_DIR SKIP_PREPARE=$SKIP_PREPARE RESUME=${RESUME:-none} WORKERS=$WORKERS" | tee -a "$LOG_FILE"

if [[ ! -f "$ROOT/$CONFIG" ]]; then
  echo "ERROR: config not found: $ROOT/$CONFIG" | tee -a "$LOG_FILE" >&2
  echo "  Set CNN_ROOT to your integration checkout, e.g.:" >&2
  echo "    CNN_ROOT=\$HOME/Codes/4IM06-G3-Project22-integration/CNN/spectral-history-cnn" >&2
  exit 1
fi

if [[ "$SKIP_PREPARE" == "1" ]]; then
  RESUME="$RESUME" bash "$ROOT/scripts/run_v1_train_npu.sh" 2>&1 | tee -a "$LOG_FILE"
else
  bash "$ROOT/scripts/run_v1_pipeline_full.sh" 2>&1 | tee -a "$LOG_FILE"
fi

echo "[$(date '+%F %T')] run_v1_gpu (NPU) done" | tee -a "$LOG_FILE"
