#!/usr/bin/env bash
# Cluster-side vc wrapper for CPU-only n6 spectrum prepare (one size per job).
# Copy next to vc_cnn_spectral_v1.sh in $CODES, or run directly on a compute node.
#
# Receives: REPO_ROOT, SIZE, JOB, PREP_WORKERS, CPU_PER_TASK
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${HOME}/Codes/4IM06-G3-Project22-integration}"
WORKER="$REPO_ROOT/scripts/prepare_n6_worker.sh"
SIZE="${SIZE:-64}"
JOB="${JOB:-n6_prepare_size${SIZE}}"
PREP_WORKERS="${PREP_WORKERS:-0}"
CPU_PER_TASK="${CPU_PER_TASK:-8}"

export REPO_ROOT SIZE PREP_WORKERS

if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: worker not found: $WORKER" >&2
  exit 1
fi

echo "vc_prepare_n6: JOB=$JOB SIZE=$SIZE PREP_WORKERS=$PREP_WORKERS CPU_PER_TASK=$CPU_PER_TASK"
echo "  REPO_ROOT=$REPO_ROOT WORKER=$WORKER"

# Copy the real `vc submit` line from vc_cnn_spectral_v1.sh; CPU-only, no NPU:
#   vc submit --job "$JOB" --cpu "$CPU_PER_TASK" --npu 0 \
#     --cmd "REPO_ROOT=$REPO_ROOT SIZE=$SIZE PREP_WORKERS=$PREP_WORKERS bash $WORKER"
if command -v vc >/dev/null 2>&1 && [[ -n "${VC_SUBMIT_CMD:-}" ]]; then
  cmd="${VC_SUBMIT_CMD//\{JOB\}/$JOB}"
  cmd="${cmd//\{WORKER\}/$WORKER}"
  cmd="${cmd//\{CPU\}/$CPU_PER_TASK}"
  cmd="${cmd//\{SIZE\}/$SIZE}"
  echo "Submitting via: $cmd"
  eval "$cmd"
else
  echo "WARN: VC_SUBMIT_CMD not set; running worker directly (interactive CPU node)." >&2
  bash "$WORKER"
fi
