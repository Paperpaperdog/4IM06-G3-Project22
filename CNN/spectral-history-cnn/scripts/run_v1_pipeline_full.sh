#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
RAISE_DIR="${RAISE_DIR:-../../spectral-mask-resampling/data/raw/raise_tiff}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
CONFIG_TAG="$(basename "${CONFIG%.yaml}")"

LOG_FILE="${LOG_DIR}/pipeline_${CONFIG_TAG}.log"
STATUS_FILE="${LOG_DIR}/pipeline_${CONFIG_TAG}.status"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
status() { echo "$1" > "$STATUS_FILE"; log "$1"; }

venv_has_torch() {
  "$1/bin/python" -c "import torch" >/dev/null 2>&1
}

activate_cpu_venv() {
  if [[ -d "$ROOT/.venv" ]] && venv_has_torch "$ROOT/.venv"; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
  elif [[ -d "$ROOT/../../spectral-mask-resampling/.venv" ]] \
    && venv_has_torch "$ROOT/../../spectral-mask-resampling/.venv"; then
    # shellcheck disable=SC1091
    source "$ROOT/../../spectral-mask-resampling/.venv/bin/activate"
  fi
}

export DEVICE="${DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export WORKERS="${WORKERS:-18}"

log "=== CNN v1 pipeline start ==="
log "config=$CONFIG raise_dir=$RAISE_DIR device=$DEVICE"

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  status "prepare"
  activate_cpu_venv
  CONFIG="$CONFIG" RAISE_DIR="$RAISE_DIR" LIMIT_SAMPLES="${LIMIT_SAMPLES:-}" WORKERS="$WORKERS" \
    bash "$ROOT/scripts/run_v1_prepare_local.sh" 2>&1 | tee -a "$LOG_FILE"
else
  log "SKIP_PREPARE=1, using existing processed data"
fi

# Do not pipe `source` — pipeline subshell drops PYTHONPATH from ensure_npu_deps.
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_python.sh" >> "$LOG_FILE" 2>&1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

python - <<'PY' | tee -a "$LOG_FILE"
from src.utils.device import print_device_info
print_device_info()
PY

OUTPUT_DIR="${OUTPUT_DIR:-$(python -c "import sys; sys.path.insert(0,'.'); from src.utils.io import load_yaml; print(load_yaml('${CONFIG}')['paths']['output_dir'])")}"
CHECKPOINT="${OUTPUT_DIR}/checkpoints/best.pt"
log "output_dir=$OUTPUT_DIR checkpoint=$CHECKPOINT"

status "train"
python src/train.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  ${RESUME:+--resume "$RESUME"} \
  2>&1 | tee -a "$LOG_FILE"

status "eval"
python src/evaluate.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --split test 2>&1 | tee -a "$LOG_FILE"

status "visualize"
python src/visualize.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CHECKPOINT" \
  --split test 2>&1 | tee -a "$LOG_FILE"

status "DONE"
log "outputs in ${OUTPUT_DIR}"
log "=== CNN v1 pipeline finished ==="
