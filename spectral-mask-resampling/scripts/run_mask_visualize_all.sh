#!/usr/bin/env bash
# Visualize learned masks for all n6 mask size sweep checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for sz in 32 64 96 128; do
  CONFIG="configs/size_sweep/n6_mask_size${sz}.yaml" \
    bash "$ROOT/scripts/run_mask_visualize_cpu.sh"
done
