#!/usr/bin/env bash
# Worker script for vc submit on GPU compute nodes.
set -euo pipefail

ROOT="/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22/CNN/spectral-history-cnn"
cd "$ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
export RAISE_DIR="${RAISE_DIR:-$ROOT/../../spectral-mask-resampling/data/raw/raise_tiff}"
export DEVICE="${DEVICE:-cuda}"
export WORKERS="${WORKERS:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vc_gpu_${TS}.log"

echo "[$(date '+%F %T')] run_v1_gpu start" | tee "$LOG_FILE"
echo "CONFIG=$CONFIG DEVICE=$DEVICE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "LIMIT_SAMPLES=${LIMIT_SAMPLES:-full} EPOCHS=${EPOCHS:-default} BATCH_SIZE=${BATCH_SIZE:-default} WORKERS=${WORKERS:-32}" | tee -a "$LOG_FILE"

bash "$ROOT/scripts/run_v1_pipeline_full.sh" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%F %T')] run_v1_gpu done" | tee -a "$LOG_FILE"
