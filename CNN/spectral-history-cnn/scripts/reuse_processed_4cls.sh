#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"
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
