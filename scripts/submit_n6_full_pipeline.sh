#!/usr/bin/env bash
# End-to-end n6 cluster workflow:
#   1) Submit CPU prepare jobs (one per observed size)
#   2) Wait until shared spectrum caches exist on REPO_ROOT
#   3) Submit Mask + CNN NPU training jobs (SKIP_PREPARE=1)
#
# Prerequisites (cluster):
#   - scripts/vc_prepare_n6.sh  -> $CODES/vc_prepare_n6.sh  (VC_SUBMIT_CMD set)
#   - spectral-mask-resampling/scripts/vc_mask.sh -> $CODES/vc_mask.sh
#   - CNN uses $CODES/vc_cnn_spectral_v1.sh (already on cluster)
#
# Usage:
#   cd 4IM06-G3-Project22
#   bash scripts/submit_n6_full_pipeline.sh
#   bash scripts/submit_n6_full_pipeline.sh --detach
#   SIZES="64 128" bash scripts/submit_n6_full_pipeline.sh
#
#   # Prepare on current node (interactive CPU), then queue NPU train:
#   bash scripts/submit_n6_full_pipeline.sh --local-prepare
#
#   # Skip prepare if caches already built:
#   SKIP_PREPARE_PHASE=1 bash scripts/submit_n6_full_pipeline.sh --train-only
#   bash scripts/submit_n6_full_pipeline.sh --prepare-only
#
# Env:
#   REPO_ROOT, CODES, SIZES, CPU_PER_TASK, PREP_WORKERS, EPOCHS
#   POLL_SEC (default 120), WAIT_TIMEOUT_SEC (0 = no timeout)
#   FORCE_PREPARE=1  rebuild caches even if present
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SIZES="${SIZES:-32 64 96 128}"
CODES="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"
CPU_PER_TASK="${CPU_PER_TASK:-16}"
PREP_WORKERS="${PREP_WORKERS:-0}"
EPOCHS="${EPOCHS:-50}"
POLL_SEC="${POLL_SEC:-120}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-0}"
SKIP_PREPARE_PHASE="${SKIP_PREPARE_PHASE:-0}"
LOCAL_PREPARE=0
DETACH=0
PREPARE_ONLY=0
TRAIN_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --local-prepare) LOCAL_PREPARE=1 ;;
    --detach) DETACH=1 ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    --train-only) TRAIN_ONLY=1; SKIP_PREPARE_PHASE=1 ;;
  esac
done

LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/n6_full_pipeline}"
mkdir -p "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/pipeline_${TS}.log"

log() {
  echo "[$(date '+%F %T')] $*"
  echo "[$(date '+%F %T')] $*" >> "$LOG_FILE"
}

cache_ready() {
  local size="$1"
  local d
  for d in \
    "$REPO_ROOT/data/processed/n6_spectra_size${size}" \
    "$(dirname "$REPO_ROOT")/data/processed/n6_spectra_size${size}"; do
    if [[ -f "$d/train_spectra.npy" && -f "$d/val_spectra.npy" && -f "$d/test_spectra.npy" ]]; then
      return 0
    fi
  done
  return 1
}

fix_legacy_cache_layout() {
  local legacy_root="$(dirname "$REPO_ROOT")/data/processed"
  local canonical_root="$REPO_ROOT/data/processed"
  if [[ -d "$legacy_root" && ! -e "$canonical_root" ]]; then
    log "Linking legacy cache $legacy_root -> $canonical_root"
    mkdir -p "$(dirname "$canonical_root")"
    ln -sfn "$legacy_root" "$canonical_root"
  fi
}

all_caches_ready() {
  local size
  for size in $SIZES; do
    if ! cache_ready "$size"; then
      return 1
    fi
  done
  return 0
}

missing_sizes() {
  local size missing=""
  for size in $SIZES; do
    if ! cache_ready "$size"; then
      missing+="$size "
    fi
  done
  echo "$missing"
}

run_prepare_phase() {
  if [[ "$SKIP_PREPARE_PHASE" == "1" ]]; then
    log "SKIP_PREPARE_PHASE=1 — skipping prepare submission"
    return 0
  fi
  if [[ "${FORCE_PREPARE:-0}" != "1" ]] && all_caches_ready; then
    log "All spectrum caches already exist — skipping prepare (FORCE_PREPARE=1 to rebuild)"
    return 0
  fi

  log "Phase 1/3: submit prepare for sizes: $SIZES"
  if [[ "$LOCAL_PREPARE" == "1" ]]; then
    PARALLEL_SIZES=1 PREP_WORKERS="$PREP_WORKERS" SIZES="$SIZES" \
      bash "$REPO_ROOT/scripts/submit_prepare_n6.sh" 2>&1 | tee -a "$LOG_FILE"
  else
    CPU_PER_TASK="$CPU_PER_TASK" PREP_WORKERS="$PREP_WORKERS" SIZES="$SIZES" \
      bash "$REPO_ROOT/scripts/submit_prepare_n6.sh" --cluster 2>&1 | tee -a "$LOG_FILE"
  fi
}

wait_for_prepare() {
  fix_legacy_cache_layout
  if all_caches_ready; then
    log "Caches ready — no wait needed"
    return 0
  fi

  log "Phase 2/3: waiting for shared caches under data/processed/n6_spectra_size*"
  log "  missing: $(missing_sizes)"
  local start_ts=$SECONDS
  while true; do
    if all_caches_ready; then
      log "All prepare caches ready."
      return 0
    fi
    if [[ "$WAIT_TIMEOUT_SEC" -gt 0 && $((SECONDS - start_ts)) -ge "$WAIT_TIMEOUT_SEC" ]]; then
      log "ERROR: timeout after ${WAIT_TIMEOUT_SEC}s; still missing sizes: $(missing_sizes)"
      exit 1
    fi
    log "  still waiting ($(missing_sizes)) — next poll in ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

verify_caches() {
  command -v python3 >/dev/null 2>&1 || return 0
  local size
  for size in $SIZES; do
    if python3 "$REPO_ROOT/scripts/verify_spectral_alignment.py" --size "$size" >> "$LOG_FILE" 2>&1; then
      log "verify OK size=$size"
    else
      log "WARN: verify failed for size=$size (see log); continuing"
    fi
  done
}

submit_train_phase() {
  log "Phase 3/3: submit Mask + CNN NPU jobs (SKIP_PREPARE=1)"

  if [[ ! -d "$CODES" ]]; then
    log "ERROR: CODES not found: $CODES — cannot submit NPU jobs" >&2
    exit 1
  fi

  export REPO_ROOT
  export CODES
  export SIZES
  export SKIP_PREPARE=1
  export EVAL_ONLY=0
  export EPOCHS

  log "Submitting Mask sweep..."
  (
    cd "$REPO_ROOT/spectral-mask-resampling"
    SKIP_PREPARE=1 SIZES="$SIZES" REPO_ROOT="$REPO_ROOT" CODES="$CODES" \
      bash scripts/submit_size_sweep_npu.sh
  ) 2>&1 | tee -a "$LOG_FILE"

  log "Submitting CNN sweep..."
  (
    cd "$REPO_ROOT/CNN/spectral-history-cnn"
    SKIP_PREPARE=1 SIZES="$SIZES" REPO_ROOT="$REPO_ROOT" CODES="$CODES" EPOCHS="$EPOCHS" \
      bash scripts/submit_size_sweep_npu.sh
  ) 2>&1 | tee -a "$LOG_FILE"

  log "Train jobs submitted. Monitor: vc list"
  log "  Mask jobs: n6_mask_size{${SIZES// /,}}"
  log "  CNN jobs:  n6_cnn_size{${SIZES// /,}}"
}

main() {
  log "=== n6 full pipeline start ==="
  log "REPO_ROOT=$REPO_ROOT CODES=$CODES SIZES=$SIZES"
  log "LOCAL_PREPARE=$LOCAL_PREPARE CPU_PER_TASK=$CPU_PER_TASK EPOCHS=$EPOCHS"

  if [[ "$TRAIN_ONLY" != "1" ]]; then
    run_prepare_phase
    fix_legacy_cache_layout
    if ! all_caches_ready; then
      wait_for_prepare
    fi
    verify_caches
  fi

  if [[ "$PREPARE_ONLY" == "1" ]]; then
    log "PREPARE_ONLY — done after prepare"
    exit 0
  fi

  fix_legacy_cache_layout

  if ! all_caches_ready; then
    log "ERROR: train phase aborted — caches missing for: $(missing_sizes)"
    exit 1
  fi

  submit_train_phase
  log "=== n6 full pipeline orchestration finished ==="
}

if [[ "$DETACH" == "1" ]]; then
  PID_FILE="$LOG_ROOT/pipeline_${TS}.pid"
  nohup env REPO_ROOT="$REPO_ROOT" CODES="$CODES" SIZES="$SIZES" \
    CPU_PER_TASK="$CPU_PER_TASK" PREP_WORKERS="$PREP_WORKERS" EPOCHS="$EPOCHS" \
    POLL_SEC="$POLL_SEC" WAIT_TIMEOUT_SEC="$WAIT_TIMEOUT_SEC" \
    SKIP_PREPARE_PHASE="$SKIP_PREPARE_PHASE" LOCAL_PREPARE="$LOCAL_PREPARE" \
    FORCE_PREPARE="${FORCE_PREPARE:-0}" \
    bash "$REPO_ROOT/scripts/submit_n6_full_pipeline.sh" \
    >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  cat <<EOF
n6 full pipeline orchestrator started in background.
  PID:  $(cat "$PID_FILE")
  Log:  $LOG_FILE

Monitor:
  tail -f $LOG_FILE

After prepare completes, Mask/CNN NPU jobs are submitted automatically.
EOF
  exit 0
fi

main
