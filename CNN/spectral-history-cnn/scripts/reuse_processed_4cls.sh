#!/usr/bin/env bash
# LEGACY (v1 4-class remap). Not part of the n6 experiment path.
# To run anyway: ALLOW_LEGACY=1 bash scripts/reuse_processed_4cls.sh
set -euo pipefail

if [[ "${ALLOW_LEGACY:-0}" != "1" ]]; then
  echo "ERROR: reuse_processed_4cls.sh is a legacy v1 utility (4-class cache remap)." >&2
  echo "  n6 experiments use configs/size_sweep/n6_poscnn_size*.yaml and fresh prepare." >&2
  echo "  See docs/EXPERIMENT_RUNBOOK.md" >&2
  echo "  To force legacy run: ALLOW_LEGACY=1 $0" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

CONFIG="${CONFIG:-configs/legacy/v1_final64_poscnn_local.yaml}"
SRC_PROCESSED_DIR="${SRC_PROCESSED_DIR:-data/processed/v1_final64_tv_rfft}"

DST_PROCESSED_DIR="$(
  CONFIG_PATH="$CONFIG" python - <<'PY'
import os
from src.utils.io import load_yaml

cfg = load_yaml(os.environ["CONFIG_PATH"])
print(cfg["paths"]["processed_dir"])
PY
)"

python scripts/remap_processed_4cls.py \
  --src-processed-dir "$SRC_PROCESSED_DIR" \
  --dst-processed-dir "$DST_PROCESSED_DIR" \
  --classes original JPEG downsample_x8 downsample_x16

echo "Remapped 4-class processed data ready at: $DST_PROCESSED_DIR"
