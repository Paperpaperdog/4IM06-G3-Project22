#!/usr/bin/env bash
# Submit or background-run n6 spectrum prepare (Mask + CNN shared cache).
#
# Local — one size:
#   SIZE=64 bash scripts/submit_prepare_n6.sh
#
# Local — four sizes in parallel (fastest on a multi-core node):
#   PARALLEL_SIZES=1 bash scripts/submit_prepare_n6.sh
#
# Local — detached (nohup, survives logout):
#   bash scripts/submit_prepare_n6.sh --detach
#   bash scripts/submit_prepare_n6.sh --detach --parallel
#
# Cluster — one vc job per size (CPU nodes; set VC_SUBMIT_CMD in vc_prepare_n6.sh):
#   CODES=/path/to/Codes bash scripts/submit_prepare_n6.sh --cluster
#   SIZES="64 128" CPU_PER_TASK=16 bash scripts/submit_prepare_n6.sh --cluster
#
# Smoke:
#   LIMIT_IMAGES=4 SAMPLES_PER_CLASS_PER_SIZE=8 SIZE=64 bash scripts/submit_prepare_n6.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SIZES="${SIZES:-${SIZE:-32 64 96 128}}"
PREP_WORKERS="${PREP_WORKERS:-0}"
PARALLEL_SIZES="${PARALLEL_SIZES:-0}"
USE_CLUSTER=0
DETACH=0

for arg in "$@"; do
  case "$arg" in
    --cluster) USE_CLUSTER=1 ;;
    --detach) DETACH=1 ;;
    --parallel) PARALLEL_SIZES=1 ;;
  esac
done

LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/prepare_n6}"
mkdir -p "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"

run_one_size() {
  local size="$1"
  local log_file="$LOG_ROOT/prepare_size${size}_${TS}.log"
  echo "=== size=$size log=$log_file ==="
  REPO_ROOT="$REPO_ROOT" SIZE="$size" PREP_WORKERS="$PREP_WORKERS" \
    LIMIT_IMAGES="${LIMIT_IMAGES:-}" \
    SAMPLES_PER_CLASS_PER_SIZE="${SAMPLES_PER_CLASS_PER_SIZE:-}" \
    bash "$REPO_ROOT/scripts/prepare_n6_worker.sh" \
    2>&1 | tee "$log_file"
}

submit_cluster_size() {
  local size="$1"
  local codes="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"
  local wrapper_name="${VC_PREPARE_WRAPPER:-vc_prepare_n6.sh}"
  local wrapper=""
  if [[ -f "$codes/$wrapper_name" ]]; then
    wrapper="$codes/$wrapper_name"
  elif [[ -f "$REPO_ROOT/scripts/vc_prepare_n6.sh" ]]; then
    wrapper="$REPO_ROOT/scripts/vc_prepare_n6.sh"
    echo "NOTE: using repo wrapper $wrapper (copy to $codes/$wrapper_name for login-node submit)" >&2
  else
    echo "ERROR: missing vc_prepare_n6.sh in $codes and $REPO_ROOT/scripts/" >&2
    exit 1
  fi
  if [[ ! -d "$codes" ]]; then
    echo "ERROR: CODES dir not found: $codes" >&2
    exit 1
  fi
  echo "=== cluster submit size=$size JOB=n6_prepare_size${size} wrapper=$wrapper ==="
  (
    cd "$codes"
    REPO_ROOT="$REPO_ROOT" \
      SIZE="$size" \
      PREP_WORKERS="$PREP_WORKERS" \
      CPU_PER_TASK="${CPU_PER_TASK:-8}" \
      JOB="n6_prepare_size${size}" \
      bash "$wrapper"
  )
}

dispatch_local() {
  local pids=()
  for size in $SIZES; do
    if [[ "$PARALLEL_SIZES" == "1" ]]; then
      run_one_size "$size" &
      pids+=("$!")
    else
      run_one_size "$size"
    fi
  done
  if [[ "$PARALLEL_SIZES" == "1" && ${#pids[@]} -gt 0 ]]; then
    echo "Waiting for ${#pids[@]} parallel prepare job(s)..."
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  fi
}

if [[ "$USE_CLUSTER" == "1" ]]; then
  for size in $SIZES; do
    submit_cluster_size "$size"
  done
  echo "Submitted prepare jobs for sizes: $SIZES"
  echo "Monitor: vc list  |  logs under $LOG_ROOT when workers tee locally"
  exit 0
fi

if [[ "$DETACH" == "1" ]]; then
  DETACH_LOG="$LOG_ROOT/submit_${TS}.log"
  PID_FILE="$LOG_ROOT/submit_${TS}.pid"
  nohup env REPO_ROOT="$REPO_ROOT" SIZES="$SIZES" PREP_WORKERS="$PREP_WORKERS" \
    PARALLEL_SIZES="$PARALLEL_SIZES" \
    LIMIT_IMAGES="${LIMIT_IMAGES:-}" \
    SAMPLES_PER_CLASS_PER_SIZE="${SAMPLES_PER_CLASS_PER_SIZE:-}" \
    bash "$REPO_ROOT/scripts/submit_prepare_n6.sh" \
    >> "$DETACH_LOG" 2>&1 &
  echo $! > "$PID_FILE"
  cat <<EOF
Prepare started in background.
  PID:      $(cat "$PID_FILE")
  Log:      $DETACH_LOG
  Per-size: $LOG_ROOT/prepare_size*_${TS}.log

Monitor:
  tail -f $DETACH_LOG
  ls -lt $LOG_ROOT/prepare_size*_${TS}.log | head
EOF
  exit 0
fi

dispatch_local
echo "Done. Verify: python scripts/verify_spectral_alignment.py --size 64"
echo "Then train with SKIP_PREPARE=1 on Mask/CNN pipelines."
