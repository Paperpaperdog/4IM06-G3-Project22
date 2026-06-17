#!/usr/bin/env bash
# Submit one CPU vc job per size to run test eval on existing CNN checkpoints.
#
#   bash scripts/submit_cnn_eval.sh
#   SIZES="64 128" bash scripts/submit_cnn_eval.sh
#
# Copy scripts/vc_cnn_eval.sh to ~/Codes/vc_cnn_eval.sh first (or set VC_SUBMIT_CMD).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$ROOT/../.." && pwd)}"
CNN_ROOT="${CNN_ROOT:-$ROOT}"
CODES="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"
SIZES="${SIZES:-32 64 96 128}"

VC_SCRIPT="${VC_SCRIPT:-$CODES/vc_cnn_eval.sh}"
if [[ ! -f "$VC_SCRIPT" && -f "$ROOT/scripts/vc_cnn_eval.sh" ]]; then
  VC_SCRIPT="$ROOT/scripts/vc_cnn_eval.sh"
fi

for size in $SIZES; do
  cfg="configs/size_sweep/n6_poscnn_size${size}.yaml"
  ckpt="$REPO_ROOT/results/cnn/n6_poscnn_size${size}/checkpoints/best.pt"
  if [[ ! -f "$ROOT/$cfg" ]]; then
    echo "SKIP: missing config $ROOT/$cfg" >&2
    continue
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP: missing checkpoint $ckpt" >&2
    continue
  fi
  echo "=== submit CNN eval size=$size ==="
  REPO_ROOT="$REPO_ROOT" \
  CNN_ROOT="$CNN_ROOT" \
  CONFIG="$cfg" \
  JOB="n6_cnn_eval_size${size}" \
  bash "$VC_SCRIPT"
done

echo "Submitted CNN eval jobs for sizes: $SIZES"
echo "After completion, re-pack or rsync results/cnn/n6_poscnn_size*/metrics.json"
