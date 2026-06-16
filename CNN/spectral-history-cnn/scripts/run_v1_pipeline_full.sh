#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
RAISE_DIR="${RAISE_DIR:-../../spectral-mask-resampling/data/raw/raise_tiff}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/pipeline_full.log"
STATUS_FILE="${LOG_DIR}/pipeline_full.status"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
status() { echo "$1" > "$STATUS_FILE"; log "$1"; }

venv_has_torch() {
  "$1/bin/python" -c "import torch" >/dev/null 2>&1
}

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
elif [[ -d "$ROOT/.venv" ]] && venv_has_torch "$ROOT/.venv"; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -d "$ROOT/../../spectral-mask-resampling/.venv" ]] \
  && venv_has_torch "$ROOT/../../spectral-mask-resampling/.venv"; then
  # shellcheck disable=SC1091
  source "$ROOT/../../spectral-mask-resampling/.venv/bin/activate"
fi

log "=== CNN v1 pipeline start ==="
log "config=$CONFIG raise_dir=$RAISE_DIR"

status "prepare"
CONFIG="$CONFIG" RAISE_DIR="$RAISE_DIR" LIMIT_SAMPLES="${LIMIT_SAMPLES:-}" \
  bash "$ROOT/scripts/run_v1_prepare_local.sh" 2>&1 | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORKERS="${WORKERS:-32}"
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

status "train"
python src/train.py \
  --config "$CONFIG" \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  ${DEVICE:+--device "$DEVICE"} \
  2>&1 | tee -a "$LOG_FILE"

status "eval"
python src/evaluate.py \
  --config "$CONFIG" \
  --checkpoint outputs/v1_final64_poscnn/checkpoints/best.pt \
  --split test 2>&1 | tee -a "$LOG_FILE"

status "visualize"
python src/visualize.py \
  --config "$CONFIG" \
  --checkpoint outputs/v1_final64_poscnn/checkpoints/best.pt \
  --split test 2>&1 | tee -a "$LOG_FILE"

status "DONE"
log "outputs in outputs/v1_final64_poscnn"
log "=== CNN v1 pipeline finished ==="
