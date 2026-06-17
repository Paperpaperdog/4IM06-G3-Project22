#!/usr/bin/env bash
# Reference cluster-side vc wrapper for the unified 6-class mask pipeline.
# This is the mask counterpart of vc_cnn_spectral_v1.sh. Copy it next to
# vc_cnn_spectral_v1.sh in your $CODES directory on the cluster.
#
# It receives env from submit_size_sweep_npu.sh (CONFIG, JOB, SKIP_PREPARE, ...)
# and submits a vc job that runs spectral-mask-resampling/scripts/vc_worker.sh
# on an Ascend NPU node.
#
# IMPORTANT: the exact `vc` submit command is cluster-specific. Copy the single
# `vc ...` submit line from your working vc_cnn_spectral_v1.sh and only swap the
# worker script path to $WORKER below. Until you do, this script falls back to
# running the worker directly (use that on an already-allocated NPU node).
set -euo pipefail

# Cluster checkout root for this repo. submit_size_sweep_npu.sh passes REPO_ROOT
# automatically; override only if you invoke this wrapper by hand.
REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "$REPO_ROOT" ]]; then
  # Fallback when run directly without submit script (edit to your checkout).
  REPO_ROOT="${HOME}/Codes/4IM06-G3-Project22-integration"
fi
MASK_ROOT="$REPO_ROOT/spectral-mask-resampling"
WORKER="$MASK_ROOT/scripts/vc_worker.sh"

export CONFIG="${CONFIG:-configs/u6_mask_combined.yaml}"
export SKIP_PREPARE="${SKIP_PREPARE:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
JOB="${JOB:-mask_u6}"
CPU_PER_TASK="${CPU_PER_TASK:-20}"

if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: mask worker not found: $WORKER" >&2
  echo "Set REPO_ROOT to the cluster checkout of this repo." >&2
  exit 1
fi

echo "vc_mask_u6: JOB=$JOB CONFIG=$CONFIG SKIP_PREPARE=$SKIP_PREPARE WORKER=$WORKER"

# ---------------------------------------------------------------------------
# TODO(cluster): replace the block below with the real `vc` submit line copied
# from vc_cnn_spectral_v1.sh, e.g. (illustrative only):
#
#   vc submit \
#     --job "$JOB" \
#     --image <ascend-torch-npu-image> \
#     --cpu "$CPU_PER_TASK" --npu 1 \
#     -- bash "$WORKER"
#
# Pass CONFIG / SKIP_PREPARE through to the worker the same way the CNN wrapper
# passes them (they are already exported above).
# ---------------------------------------------------------------------------
if command -v vc >/dev/null 2>&1 && [[ -n "${VC_SUBMIT_CMD:-}" ]]; then
  # Optional: provide a full submit command template via VC_SUBMIT_CMD, e.g.
  #   VC_SUBMIT_CMD='vc submit --job {JOB} --npu 1 -- bash {WORKER}'
  cmd="${VC_SUBMIT_CMD//\{JOB\}/$JOB}"
  cmd="${cmd//\{WORKER\}/$WORKER}"
  cmd="${cmd//\{CPU\}/$CPU_PER_TASK}"
  echo "Submitting via: $cmd"
  eval "$cmd"
else
  echo "WARN: no vc submit command configured; running worker directly." >&2
  echo "      (fine on an allocated NPU node; set VC_SUBMIT_CMD to queue a job)" >&2
  bash "$WORKER"
fi
