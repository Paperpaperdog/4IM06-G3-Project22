#!/usr/bin/env bash
# Full mask pipeline for ONE config: prepare (optional) -> train -> eval -> viz.
# Output dir and checkpoint are derived from the config, so this works for the
# combined config and every per-size sweep config without edits.
#
# Examples:
#   CONFIG=configs/size_sweep/u6_mask_size64.yaml bash scripts/run_pipeline_config.sh
#   SKIP_PREPARE=1 CONFIG=configs/u6_mask_combined.yaml bash scripts/run_pipeline_config.sh
#   LIMIT_IMAGES=4 SAMPLES_PER_CLASS_PER_SIZE=8 CONFIG=... bash scripts/run_pipeline_config.sh  # smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:?Set CONFIG=configs/size_sweep/u6_mask_size64.yaml}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/mask_pipeline_${TS}.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }

DEVICE_NAME="$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['training']['device'])" 2>/dev/null || echo npu)"

log "=== mask pipeline start ==="
log "CONFIG=$CONFIG DEVICE=$DEVICE_NAME SKIP_PREPARE=$SKIP_PREPARE"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_python.sh" cpu
  PREP_PY="${PYTHON:-python3}"
  log "prepare python: $PREP_PY"
  CONFIG="$CONFIG" PYTHON="$PREP_PY" bash "$ROOT/scripts/run_prepare_config.sh" 2>&1 | tee -a "$LOG_FILE"
else
  log "SKIP_PREPARE=1, using existing cache"
fi

if [[ "$DEVICE_NAME" == "npu" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_python.sh" npu
else
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_python.sh" cpu
fi
PY="${PYTHON:-python3}"
log "train/eval python: $PY"

OUTPUT_DIR="$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
CKPT="$OUTPUT_DIR/checkpoints/best.pt"
log "OUTPUT_DIR=$OUTPUT_DIR"

log "train"
"$PY" src/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

log "eval"
"$PY" src/evaluate.py --config "$CONFIG" --checkpoint "$CKPT" --split test 2>&1 | tee -a "$LOG_FILE"

log "visualize"
"$PY" src/visualize.py --config "$CONFIG" --checkpoint "$CKPT" 2>&1 | tee -a "$LOG_FILE" || \
  log "WARN: visualize skipped (optional plotting deps missing)"

log "=== mask pipeline done -> $OUTPUT_DIR ==="
