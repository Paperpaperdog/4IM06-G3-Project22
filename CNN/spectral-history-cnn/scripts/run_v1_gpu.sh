#!/usr/bin/env bash
# Worker for vc submit on Ascend NPU nodes (pdgpu-sjtu-ai).
set -euo pipefail

ROOT="/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22/CNN/spectral-history-cnn"
cd "$ROOT"

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

TS="$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vc_npu_${TS}.log"

echo "[$(date '+%F %T')] run_v1_gpu (NPU) start" | tee "$LOG_FILE"
echo "CONFIG=$CONFIG DEVICE=$DEVICE ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "SKIP_PREPARE=$SKIP_PREPARE RESUME=${RESUME:-none} WORKERS=$WORKERS" | tee -a "$LOG_FILE"

if [[ "$SKIP_PREPARE" == "1" ]]; then
  RESUME="$RESUME" bash "$ROOT/scripts/run_v1_train_npu.sh" 2>&1 | tee -a "$LOG_FILE"
else
  bash "$ROOT/scripts/run_v1_pipeline_full.sh" 2>&1 | tee -a "$LOG_FILE"
fi

echo "[$(date '+%F %T')] run_v1_gpu (NPU) done" | tee -a "$LOG_FILE"
