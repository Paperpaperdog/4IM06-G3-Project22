#!/usr/bin/env bash
# Run the unified 6-class spectral mask across all observed sizes, sequentially,
# in the current shell (interactive GPU/NPU node or local smoke test).
#
#   bash scripts/run_size_sweep.sh
#   SIZES="64 128" bash scripts/run_size_sweep.sh
#   LIMIT_IMAGES=4 SAMPLES_PER_CLASS_PER_SIZE=8 bash scripts/run_size_sweep.sh  # smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZES="${SIZES:-32 64 96 128}"
SWEEP_TAG="${SWEEP_TAG:-u6}"

for size in $SIZES; do
  cfg="configs/size_sweep/${SWEEP_TAG}_mask_size${size}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "SKIP: missing config $cfg" >&2
    continue
  fi
  echo "=== Mask size sweep: size=$size config=$cfg ==="
  CONFIG="$cfg" bash "$ROOT/scripts/run_pipeline_config.sh"
done

echo "All mask size-sweep runs finished (SWEEP_TAG=$SWEEP_TAG). Summarize with:"
echo "  python ../scripts/analysis/summarize_size_effect.py --variant $SWEEP_TAG"
