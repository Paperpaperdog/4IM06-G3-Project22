#!/usr/bin/env bash
# Test-set eval + visualize only (no train, no prepare).
# Use on a cluster CPU node with the mask project venv (sklearn/pandas/matplotlib).
#
#   CONFIG=configs/size_sweep/n6_poscnn_size64.yaml bash scripts/run_v1_eval_cpu.sh
set -euo pipefail

ROOT="${CNN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export CNN_ROOT="$ROOT"
export PYTHONPATH=.
export DEVICE="${DEVICE:-cpu}"

CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"
SPLIT="${SPLIT:-test}"
repo_root="$(cd "$ROOT/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${repo_root/-integration/}/spectral-mask-resampling/.venv}"

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_cpu_${TS:-$(date +%Y%m%d_%H%M%S)}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

if [[ ! -f "$ROOT/$CONFIG" ]]; then
  echo "ERROR: config not found: $ROOT/$CONFIG" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found: $VENV_DIR/bin/python" >&2
  exit 1
fi

PY="$VENV_DIR/bin/python"
OUTPUT_DIR="$("$PY" -c "import yaml; print(yaml.safe_load(open('$ROOT/$CONFIG'))['paths']['output_dir'])")"
CKPT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/best.pt}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

log "=== CNN eval-only (CPU) ==="
log "config=$CONFIG split=$SPLIT device=$DEVICE"
log "checkpoint=$CKPT output_dir=$OUTPUT_DIR"
log "python=$PY"

"$PY" src/evaluate.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CKPT" \
  --split "$SPLIT" 2>&1 | tee -a "$LOG_FILE"

if [[ "${SKIP_VISUALIZE:-0}" == "1" ]]; then
  log "SKIP_VISUALIZE=1 (metrics-only)"
elif "$PY" src/visualize.py \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --checkpoint "$CKPT" \
  --split "$SPLIT" \
  --num-workers 0 2>&1 | tee -a "$LOG_FILE"; then
  log "visualize OK"
else
  log "WARN: visualize skipped (metrics already saved)"
fi

log "=== done: $OUTPUT_DIR/metrics.json ==="
