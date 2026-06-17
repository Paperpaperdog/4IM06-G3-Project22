#!/usr/bin/env bash
# Train + eval + visualize on NPU (skip prepare if data already exists).
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export CNN_ROOT="$ROOT"

CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_npu_${TS:-$(date +%Y%m%d_%H%M%S)}.log"
STATUS_FILE="${LOG_DIR}/pipeline_full.status"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
status() { echo "$1" > "$STATUS_FILE"; log "$1"; }

export DEVICE="${DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

# Do not pipe `source` — pipeline subshell drops PYTHONPATH from ensure_npu_deps.
# shellcheck disable=SC1091
source "$ROOT/scripts/activate_python.sh" >> "$LOG_FILE" 2>&1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

log "=== CNN NPU train-only start ==="
log "config=$CONFIG device=$DEVICE ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

PY="${PYTHON:-python3}"
"$PY" - <<'PY' | tee -a "$LOG_FILE"
from src.utils.device import print_device_info
print_device_info()
PY

# Derive output dir (and checkpoint) from the config for sweep compatibility.
OUTPUT_DIR="$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['paths']['output_dir'])")"
CKPT="$OUTPUT_DIR/checkpoints/best.pt"
log "OUTPUT_DIR=$OUTPUT_DIR"

status "train"
"$PY" src/train.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
  ${RESUME:+--resume "$RESUME"} \
  2>&1 | tee -a "$LOG_FILE"

status "eval"
if "$PY" src/evaluate.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CKPT" \
  --split test 2>&1 | tee -a "$LOG_FILE"; then
  status "visualize"
  "$PY" src/visualize.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --checkpoint "$CKPT" \
    --split test 2>&1 | tee -a "$LOG_FILE"
else
  log "WARN: eval/visualize skipped (optional deps missing on NPU image)"
fi

status "DONE"
log "=== CNN NPU train-only finished ==="
