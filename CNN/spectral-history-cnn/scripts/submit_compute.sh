#!/usr/bin/env bash
# Submit CNN full pipeline on a compute node (local RAISE TIFF from spectral-mask-resampling).
#
# Recommended (GPU compute node via vc):
#   cd /aistor/sjtu/hpc_stor01/home/jinbingrui/Codes
#   bash vc_cnn_spectral_v1.sh
#
# GPU smoke test:
#   LIMIT_SAMPLES=100 EPOCHS=5 BATCH_SIZE=64 bash vc_cnn_spectral_v1.sh
#
# Fallback (nohup on current login/debug node):
#   cd CNN/spectral-history-cnn
#   bash scripts/submit_compute.sh
#
# Debug (small sample count, no vc):
#   LIMIT_SAMPLES=100 EPOCHS=5 BATCH_SIZE=64 DEVICE=cpu bash scripts/submit_compute.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
export RAISE_DIR="${RAISE_DIR:-$ROOT/../../spectral-mask-resampling/data/raw/raise_tiff}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"
export EPOCHS="${EPOCHS:-}"
export BATCH_SIZE="${BATCH_SIZE:-}"
export DEVICE="${DEVICE:-cuda}"

TS="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="logs/submit_${TS}.nohup.log"

if [[ ! -d "$RAISE_DIR" ]]; then
  echo "ERROR: RAISE directory missing: $RAISE_DIR" >&2
  echo "Set RAISE_DIR to spectral-mask-resampling/data/raw/raise_tiff" >&2
  exit 1
fi

activate_venv() {
  # shellcheck disable=SC1091
  source "$1/bin/activate"
}

venv_has_torch() {
  "$1/bin/python" -c "import torch" >/dev/null 2>&1
}

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
FALLBACK_VENV="$ROOT/../../spectral-mask-resampling/.venv"

if [[ -n "${PYTHON:-}" ]] && venv_has_torch "$(dirname "$(dirname "$PYTHON")")"; then
  echo "Using existing PYTHON=$PYTHON"
elif [[ -d "$VENV_DIR" ]] && venv_has_torch "$VENV_DIR"; then
  echo "Using venv: $VENV_DIR"
  activate_venv "$VENV_DIR"
elif [[ -d "$FALLBACK_VENV" ]] && venv_has_torch "$FALLBACK_VENV"; then
  echo "Using fallback venv (spectral-mask-resampling): $FALLBACK_VENV"
  activate_venv "$FALLBACK_VENV"
else
  echo "Creating venv and installing deps (may take several minutes on shared storage)..."
  python3 -m venv "$VENV_DIR"
  activate_venv "$VENV_DIR"
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo "submit CNN pipeline $(date -Is)" | tee "$NOHUP_LOG"
echo "RAISE_DIR=$RAISE_DIR" | tee -a "$NOHUP_LOG"
echo "CONFIG=$CONFIG CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$NOHUP_LOG"
echo "LIMIT_SAMPLES=${LIMIT_SAMPLES:-full} EPOCHS=${EPOCHS:-default} BATCH_SIZE=${BATCH_SIZE:-default}" | tee -a "$NOHUP_LOG"

nohup bash scripts/run_v1_pipeline_full.sh >> "$NOHUP_LOG" 2>&1 &
PID=$!
echo "$PID" > logs/pipeline.pid

echo "SUBMITTED PID=$PID"
echo "monitor:  tail -f logs/pipeline_full.log"
echo "status:   cat logs/pipeline_full.status"
echo "nohup:    tail -f $NOHUP_LOG"
echo "For GPU compute node, prefer: bash /aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/vc_cnn_spectral_v1.sh"
