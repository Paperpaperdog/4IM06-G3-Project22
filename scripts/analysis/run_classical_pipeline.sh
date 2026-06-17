#!/usr/bin/env bash
# Classical route A: forensic dataset → DCT-FFT size sweep → (optional) NFA sweep
# → (optional) unified 3-method comparison.
#
# Foreground (watch progress in terminal + log):
#   cd 4IM06-G3-Project22
#   bash scripts/analysis/run_classical_pipeline.sh
#
# Background (detached, progress via tail):
#   bash scripts/analysis/run_classical_pipeline.sh --detach
#   tail -f test_results/classical_pipeline_logs/latest/pipeline.log
#   cat  test_results/classical_pipeline_logs/latest/progress.txt
#
# Options via env:
#   SKIP_NFA=1              skip classical_size_sweep (step 3)
#   SKIP_UNIFIED=1          skip unified_method_comparison (step 4)
#   LIMIT_IMAGES=20         passed to classical_size_sweep
#   WORKERS=0               parallel workers (0 = all cores)
#   FORENSIC_INPUT=...      default: spectral-mask-resampling/data/raw/raise_tiff
#   FORENSIC_OUT=...        default: test_results/forensic_pp
#   FORENSIC_LIMIT=100      cap step-1 images (omit = all TIFFs)
#   CLASS_SET=u7            7-class classical (+ upsample_x8 in forensic dataset)
#   CLASS_SET=n6            native-spectrum sweep set (upsample factors 4,8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Detach: re-launch under nohup and exit immediately with monitor hints.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--detach" ]]; then
  shift
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  LOG_ROOT="${LOG_ROOT:-test_results/classical_pipeline_logs}"
  LOG_DIR="$LOG_ROOT/$RUN_ID"
  mkdir -p "$LOG_DIR"
  ln -sfn "$RUN_ID" "$LOG_ROOT/latest"
  MAIN_LOG="$LOG_DIR/pipeline.log"
  PID_FILE="$LOG_DIR/pipeline.pid"

  nohup env RUN_ID="$RUN_ID" LOG_DIR="$LOG_DIR" DETACHED=1 PYTHONUNBUFFERED=1 \
    bash "$SCRIPT_DIR/run_classical_pipeline.sh" "$@" \
    >> "$MAIN_LOG" 2>&1 &
  echo $! > "$PID_FILE"

  cat <<EOF
Classical pipeline started in background.
  PID:      $(cat "$PID_FILE")
  Log dir:  $LOG_DIR
  Main log: $MAIN_LOG
  Progress: $LOG_DIR/progress.txt
  Status:   $LOG_DIR/status.txt

Monitor:
  tail -f $MAIN_LOG
  watch -n 30 cat $LOG_DIR/progress.txt

Stop:
  kill $(cat "$PID_FILE")
EOF
  exit 0
fi

# ---------------------------------------------------------------------------
# Config (overridable via environment)
# ---------------------------------------------------------------------------
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-test_results/classical_pipeline_logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"
mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_DIR")" "$LOG_ROOT/latest" 2>/dev/null || true

MAIN_LOG="$LOG_DIR/pipeline.log"
PROGRESS_FILE="$LOG_DIR/progress.txt"
STATUS_FILE="$LOG_DIR/status.txt"
PID_FILE="$LOG_DIR/pipeline.pid"
echo $$ > "$PID_FILE"

FORENSIC_INPUT="${FORENSIC_INPUT:-spectral-mask-resampling/data/raw/raise_tiff}"
NFA_OUT="${NFA_OUT:-test_results/classical_size_sweep}"
MAX_SIZES="${MAX_SIZES:-32,64,96,128}"
LIMIT_IMAGES="${LIMIT_IMAGES:-20}"
WORKERS="${WORKERS:-0}"
SKIP_NFA="${SKIP_NFA:-0}"
SKIP_UNIFIED="${SKIP_UNIFIED:-0}"
FORENSIC_LIMIT="${FORENSIC_LIMIT:-}"
CLASS_SET="${CLASS_SET:-u6}"

if [[ "$CLASS_SET" == "u7" ]]; then
  FORENSIC_OUT="${FORENSIC_OUT:-test_results/forensic_pp_u7}"
  JPEG_SWEEP_OUT="${JPEG_SWEEP_OUT:-test_results/jpeg_detector_size_sweep_u7}"
  UNIFIED_OUT="${UNIFIED_OUT:-test_results/unified_comparison_u7}"
  FORENSIC_UPSAMPLE_FACTORS="${FORENSIC_UPSAMPLE_FACTORS:-2,4,8}"
elif [[ "$CLASS_SET" == "n6" ]]; then
  FORENSIC_OUT="${FORENSIC_OUT:-test_results/forensic_pp_n6}"
  JPEG_SWEEP_OUT="${JPEG_SWEEP_OUT:-test_results/jpeg_detector_size_sweep_n6}"
  UNIFIED_OUT="${UNIFIED_OUT:-test_results/unified_comparison_n6}"
  FORENSIC_UPSAMPLE_FACTORS="${FORENSIC_UPSAMPLE_FACTORS:-4,8}"
else
  FORENSIC_OUT="${FORENSIC_OUT:-test_results/forensic_pp}"
  JPEG_SWEEP_OUT="${JPEG_SWEEP_OUT:-test_results/jpeg_detector_size_sweep}"
  UNIFIED_OUT="${UNIFIED_OUT:-test_results/unified_comparison}"
  FORENSIC_UPSAMPLE_FACTORS="${FORENSIC_UPSAMPLE_FACTORS:-2,4}"
fi

export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

PIPELINE_START=$(date +%s)
STEP_TIMES=()

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  if [[ "${DETACHED:-}" == "1" ]]; then
    echo "$msg"
  else
    echo "$msg" | tee -a "$MAIN_LOG"
  fi
}

format_duration() {
  local secs=$1
  local h=$((secs / 3600))
  local m=$(((secs % 3600) / 60))
  local s=$((secs % 60))
  if (( h > 0 )); then
    printf '%dh%02dm%02ds' "$h" "$m" "$s"
  elif (( m > 0 )); then
    printf '%dm%02ds' "$m" "$s"
  else
    printf '%ds' "$s"
  fi
}

estimate_eta_seconds() {
  local done=$1
  local total=$2
  if (( done <= 0 || done >= total )); then
    echo 0
    return
  fi
  local elapsed=$(( $(date +%s) - PIPELINE_START ))
  # avg seconds per completed step × remaining steps
  echo $(( elapsed * (total - done) / done ))
}

write_progress() {
  local step_idx=$1
  local step_name=$2
  local step_state=$3   # running | done | skipped
  local step_elapsed=${4:-0}
  local total_steps=$5
  local eta_s
  eta_s=$(estimate_eta_seconds "$(( step_idx - 1 ))" "$total_steps")
  local eta_str
  if [[ "$step_state" == "running" && "$eta_s" -gt 0 ]]; then
    local finish_ts=$(( $(date +%s) + eta_s ))
    local finish_clock
    finish_clock="$(date -d "@$finish_ts" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$finish_ts" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'N/A')"
    eta_str="$(format_duration "$eta_s") (est. finish ~$finish_clock)"
  else
    eta_str="—"
  fi
  {
    echo "run_id:        $RUN_ID"
    echo "pid:           $$"
    echo "updated:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "step:          $step_idx / $total_steps"
    echo "current:       $step_name ($step_state)"
    echo "elapsed_total: $(format_duration "$(( $(date +%s) - PIPELINE_START ))")"
    if [[ "$step_state" == "running" ]]; then
      echo "step_elapsed:  $(format_duration "$step_elapsed")"
    fi
    echo "eta_remaining: $eta_str"
    echo ""
    echo "log:      $MAIN_LOG"
    echo "status:   $STATUS_FILE"
  } > "$PROGRESS_FILE"
  echo "$step_name:$step_state" > "$STATUS_FILE"
}

resolve_raise_tiff() {
  local p="$1"
  local legacy="${PROJECT_ROOT/-integration/}"
  for candidate in \
    "$PROJECT_ROOT/$p" \
    "$legacy/spectral-mask-resampling/data/raw/raise_tiff"
  do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$PROJECT_ROOT/$p"
}

run_step() {
  local step_idx=$1
  local step_name=$2
  local total_steps=$3
  shift 3
  local step_log="$LOG_DIR/step_${step_idx}_${step_name}.log"

  log "======== STEP $step_idx/$total_steps: $step_name ========"
  write_progress "$step_idx" "$step_name" "running" 0 "$total_steps"

  local t0=$(date +%s)
  set +e
  if [[ "${DETACHED:-}" == "1" ]]; then
    "$@" 2>&1 | tee -a "$step_log"
  else
    "$@" 2>&1 | tee -a "$step_log" | tee -a "$MAIN_LOG"
  fi
  local rc=${PIPESTATUS[0]}
  set -e
  local t1=$(date +%s)
  local dt=$((t1 - t0))
  STEP_TIMES+=("$dt")

  if [[ "$rc" -ne 0 ]]; then
    log "ERROR: step '$step_name' failed (exit $rc). See $step_log"
    write_progress "$step_idx" "$step_name" "FAILED" "$dt" "$total_steps"
    exit "$rc"
  fi

  log "STEP $step_idx done in $(format_duration "$dt")"
  write_progress "$step_idx" "$step_name" "done" "$dt" "$total_steps"
}

# Count planned steps
TOTAL_STEPS=2
[[ "$SKIP_NFA" != "1" ]] && TOTAL_STEPS=$((TOTAL_STEPS + 1))
[[ "$SKIP_UNIFIED" != "1" ]] && TOTAL_STEPS=$((TOTAL_STEPS + 1))

RAISE_DIR="$(resolve_raise_tiff "$FORENSIC_INPUT")"

log "=== Classical pipeline start ==="
log "PROJECT_ROOT=$PROJECT_ROOT"
log "LOG_DIR=$LOG_DIR"
log "RAISE_DIR=$RAISE_DIR"
log "FORENSIC_OUT=$FORENSIC_OUT MAX_SIZES=$MAX_SIZES WORKERS=$WORKERS"
log "SKIP_NFA=$SKIP_NFA SKIP_UNIFIED=$SKIP_UNIFIED LIMIT_IMAGES=$LIMIT_IMAGES CLASS_SET=$CLASS_SET"
log "FORENSIC_UPSAMPLE_FACTORS=$FORENSIC_UPSAMPLE_FACTORS"
log "Planned steps: $TOTAL_STEPS"

if [[ ! -d "$RAISE_DIR" ]]; then
  log "ERROR: RAISE TIFF dir not found: $RAISE_DIR"
  exit 1
fi

STEP=1

# Step 1: forensic post-process dataset
_forensic_args=(
  "$PY" -u create_forensic_postprocess_dataset.py
  --input_dir "$RAISE_DIR"
  --output_dir "$FORENSIC_OUT"
  --include_original
  --include_upsampling
  --mix_order both
  --upsample_factors "$FORENSIC_UPSAMPLE_FACTORS"
)
if [[ -n "$FORENSIC_LIMIT" ]]; then
  _forensic_args+=(--limit "$FORENSIC_LIMIT")
fi
run_step "$STEP" "forensic_dataset" "$TOTAL_STEPS" "${_forensic_args[@]}"
STEP=$((STEP + 1))

# Step 2: DCT-FFT jpeg detector size sweep
run_step "$STEP" "jpeg_detector_sweep" "$TOTAL_STEPS" \
  "$PY" scripts/analysis/jpeg_detector_size_sweep.py \
    --dataset-root "$FORENSIC_OUT" \
    --null-dir "$FORENSIC_OUT/original" \
    --max-sizes "$MAX_SIZES" \
    --workers "$WORKERS" \
    --outdir "$JPEG_SWEEP_OUT"
STEP=$((STEP + 1))

# Step 3: NFA source-size recovery (optional)
if [[ "$SKIP_NFA" != "1" ]]; then
  run_step "$STEP" "nfa_size_sweep" "$TOTAL_STEPS" \
    "$PY" scripts/analysis/classical_size_sweep.py \
      --image-dir "$RAISE_DIR" \
      --limit-images "$LIMIT_IMAGES" \
      --target-sizes "$MAX_SIZES" \
      --workers "$WORKERS" \
      --outdir-root "$NFA_OUT"
  STEP=$((STEP + 1))
else
  log "SKIP step $STEP: nfa_size_sweep (SKIP_NFA=1)"
fi

# Step 4: unified comparison (optional; skips missing Mask/CNN metrics)
if [[ "$SKIP_UNIFIED" != "1" ]]; then
  run_step "$STEP" "unified_comparison" "$TOTAL_STEPS" \
    "$PY" scripts/analysis/unified_method_comparison.py \
      --sizes "$MAX_SIZES" \
      --variant "$CLASS_SET" \
      --classical-eval-dir "$JPEG_SWEEP_OUT" \
      --outdir "$UNIFIED_OUT"
else
  log "SKIP step $STEP: unified_comparison (SKIP_UNIFIED=1)"
fi

TOTAL_ELAPSED=$(( $(date +%s) - PIPELINE_START ))
{
  echo "run_id:        $RUN_ID"
  echo "pid:           $$"
  echo "updated:       $(date '+%Y-%m-%d %H:%M:%S')"
  echo "step:          $TOTAL_STEPS / $TOTAL_STEPS"
  echo "current:       ALL DONE"
  echo "elapsed_total: $(format_duration "$TOTAL_ELAPSED")"
  echo "eta_remaining: —"
} > "$PROGRESS_FILE"
echo "DONE" > "$STATUS_FILE"

log "=== Classical pipeline finished in $(format_duration "$TOTAL_ELAPSED") ==="
log "Outputs:"
log "  forensic dataset: $FORENSIC_OUT"
log "  jpeg sweep:       $JPEG_SWEEP_OUT"
[[ "$SKIP_NFA" != "1" ]] && log "  NFA sweep:        $NFA_OUT"
[[ "$SKIP_UNIFIED" != "1" ]] && log "  unified compare:  $UNIFIED_OUT"
log "Logs: $LOG_DIR"
