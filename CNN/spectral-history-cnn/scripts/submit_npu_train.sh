#!/usr/bin/env bash
# Submit NPU training for current n6 config (default: size 64).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="${CODES:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes}"

cd "$CODES"
SKIP_PREPARE="${SKIP_PREPARE:-1}" \
CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}" \
RESUME="${RESUME:-}" \
EPOCHS="${EPOCHS:-50}" \
JOB="${JOB:-n6_cnn}" \
bash vc_cnn_spectral_v1.sh
