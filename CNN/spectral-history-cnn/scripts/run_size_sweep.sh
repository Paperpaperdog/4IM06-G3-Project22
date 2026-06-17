#!/usr/bin/env bash
# Run the unified 6-class CNN across all observed sizes, sequentially, in the
# current shell (use on an interactive GPU/NPU node or locally for smoke tests).
# For one-job-per-size cluster submission use submit_size_sweep_npu.sh instead.
#
#   bash scripts/run_size_sweep.sh
#   SIZES="64 128" bash scripts/run_size_sweep.sh
#   LIMIT_SAMPLES=50 EPOCHS=2 DEVICE=cpu bash scripts/run_size_sweep.sh   # smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZES="${SIZES:-32 64 96 128}"
export DEVICE="${DEVICE:-npu}"

for size in $SIZES; do
  cfg="configs/size_sweep/u6_poscnn_size${size}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "SKIP: missing config $cfg" >&2
    continue
  fi
  echo "=== CNN size sweep: size=$size config=$cfg ==="
  CONFIG="$cfg" bash "$ROOT/scripts/run_v1_pipeline_full.sh"
done

echo "All CNN size-sweep runs finished. Summarize with:"
echo "  python ../../scripts/analysis/summarize_size_effect.py"
