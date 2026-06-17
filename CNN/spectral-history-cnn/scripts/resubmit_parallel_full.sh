#!/usr/bin/env bash
# Cancel slow single-thread preprocess job (if any) and submit full parallel pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

OLD_JOB="${OLD_JOB:-job-178156654846153226537-jinbingrui}"
WORKERS="${WORKERS:-18}"
CPU_PER_TASK="${CPU_PER_TASK:-20}"
JOB="${JOB:-n6_cnn_full_parallel}"
CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/resubmit_parallel.log"; }

if vc list 2>/dev/null | rg -q "$OLD_JOB"; then
  log "Cancel old job: $OLD_JOB"
  vc delete --job "$OLD_JOB" || true
  sleep 5
fi

log "Submit parallel full run WORKERS=$WORKERS CPU_PER_TASK=$CPU_PER_TASK JOB=$JOB CONFIG=$CONFIG"
cd "$CODES"
WORKERS="$WORKERS" CPU_PER_TASK="$CPU_PER_TASK" JOB="$JOB" CONFIG="$CONFIG" \
  bash vc_cnn_spectral_v1.sh | tee -a "$LOG_DIR/resubmit_parallel.log"
