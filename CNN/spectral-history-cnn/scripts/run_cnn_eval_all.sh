#!/usr/bin/env bash
# Run test eval for all n6 CNN sizes in the current shell (no vc submit).
# Use on a login/debug node when data + venv are already on shared storage.
#
#   cd CNN/spectral-history-cnn
#   bash scripts/run_cnn_eval_all.sh
#   SIZES="64" bash scripts/run_cnn_eval_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZES="${SIZES:-32 64 96 128}"

for size in $SIZES; do
  cfg="configs/size_sweep/n6_poscnn_size${size}.yaml"
  ckpt="$(cd "$ROOT/../.." && pwd)/results/cnn/n6_poscnn_size${size}/checkpoints/best.pt"
  if [[ ! -f "$ROOT/$cfg" ]]; then
    echo "SKIP: missing $ROOT/$cfg" >&2
    continue
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP: missing checkpoint $ckpt" >&2
    continue
  fi
  echo "=== CNN eval size=$size (CPU, local) ==="
  CONFIG="$cfg" DEVICE=cpu SKIP_VISUALIZE="${SKIP_VISUALIZE:-1}" bash "$ROOT/scripts/run_v1_eval_cpu.sh"
done

echo "Done. Check: ls $ROOT/../../results/cnn/n6_poscnn_size*/metrics.json"
