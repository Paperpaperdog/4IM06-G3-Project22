#!/usr/bin/env bash
# Export learned masks/references + mean spectra from an existing checkpoint.
# CPU-only; use on a debug/login node (no NPU required).
#
#   CONFIG=configs/size_sweep/n6_mask_size128.yaml bash scripts/run_mask_visualize_cpu.sh
#   bash scripts/run_mask_visualize_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

CONFIG="${CONFIG:?Set CONFIG=configs/size_sweep/n6_mask_size128.yaml}"

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/mask_visualize_cpu_${TS:-$(date +%Y%m%d_%H%M%S)}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

if [[ ! -f "$ROOT/$CONFIG" ]]; then
  echo "ERROR: config not found: $ROOT/$CONFIG" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/resolve_python.sh" cpu
PY="${PYTHON:-python3}"

OUTPUT_DIR="$("$PY" -c "import yaml; print(yaml.safe_load(open('$ROOT/$CONFIG'))['output_dir'])")"
CKPT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/best.pt}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

log "=== mask visualize (CPU) ==="
log "config=$CONFIG"
log "checkpoint=$CKPT"
log "output_dir=$OUTPUT_DIR"
log "python=$PY"

"$PY" src/visualize.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" 2>&1 | tee -a "$LOG_FILE"

log "=== done: $OUTPUT_DIR/figures/masks/ ==="
