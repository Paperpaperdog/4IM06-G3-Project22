#!/usr/bin/env bash
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export CNN_ROOT="$ROOT"

CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
RAISE_DIR="${RAISE_DIR:-../../spectral-mask-resampling/data/raw/raise_tiff}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/pipeline_full.log"
STATUS_FILE="${LOG_DIR}/pipeline_full.status"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
status() { echo "$1" > "$STATUS_FILE"; log "$1"; }

export DEVICE="${DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export WORKERS="${WORKERS:-18}"

log "=== CNN v1 pipeline start ==="
log "config=$CONFIG raise_dir=$RAISE_DIR device=$DEVICE"

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  status "prepare"
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_prepare_python.sh" 2>&1 | tee -a "$LOG_FILE"
  CONFIG="$CONFIG" RAISE_DIR="$RAISE_DIR" LIMIT_SAMPLES="${LIMIT_SAMPLES:-}" WORKERS="$WORKERS" \
    PREP_PY="$PREP_PY" bash "$ROOT/scripts/run_v1_prepare_local.sh" 2>&1 | tee -a "$LOG_FILE"
else
  log "SKIP_PREPARE=1, using existing processed data"
fi

export DEVICE="${DEVICE:-npu}"
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_python.sh" >> "$LOG_FILE" 2>&1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

python - <<'PY' | tee -a "$LOG_FILE"
from src.utils.device import print_device_info
print_device_info()
PY

# Derive the output dir (and thus checkpoint path) from the config so the same
# pipeline works for any config in the input-size sweep.
OUTPUT_DIR="$(python -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['paths']['output_dir'])")"
CKPT="$OUTPUT_DIR/checkpoints/best.pt"
log "OUTPUT_DIR=$OUTPUT_DIR"

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
  --checkpoint "$CKPT" \
  --split test 2>&1 | tee -a "$LOG_FILE"

status "visualize"
python src/visualize.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CKPT" \
  --split test 2>&1 | tee -a "$LOG_FILE"

status "DONE"
log "outputs in $OUTPUT_DIR"
log "=== CNN v1 pipeline finished ==="
