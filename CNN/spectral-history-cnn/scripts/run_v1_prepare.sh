#!/usr/bin/env bash
# Prepare spectra cache only (n6). Prefer run_v1_pipeline_full.sh for the full pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"
exec bash "$ROOT/scripts/run_v1_prepare_local.sh"
