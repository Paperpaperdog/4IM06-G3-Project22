#!/usr/bin/env bash
# On-node entrypoint for the mask pipeline on an Ascend NPU compute node.
# Analogous to CNN/spectral-history-cnn/scripts/run_v1_gpu.sh. A cluster vc
# wrapper (e.g. vc_mask_u6.sh) should set CONFIG and exec this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

export CONFIG="${CONFIG:-configs/u6_mask_combined.yaml}"
export SKIP_PREPARE="${SKIP_PREPARE:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vc_mask_${TS}.log"

echo "[$(date '+%F %T')] mask vc_worker start CONFIG=$CONFIG" | tee "$LOG_FILE"
bash "$ROOT/scripts/run_pipeline_config.sh" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] mask vc_worker done" | tee -a "$LOG_FILE"
