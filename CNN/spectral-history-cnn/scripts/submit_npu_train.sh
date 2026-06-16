#!/usr/bin/env bash
# Submit NPU training for current config.
# Default does NOT resume from old checkpoint (avoid class-count mismatch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes"

cd "$CODES"
SKIP_PREPARE="${SKIP_PREPARE:-1}" \
CONFIG="${CONFIG:-$ROOT/configs/v1_final64_poscnn_local.yaml}" \
RESUME="${RESUME:-}" \
EPOCHS="${EPOCHS:-50}" \
JOB="${JOB:-cnn_spectral_v1_npu_4cls}" \
bash vc_cnn_spectral_v1.sh
