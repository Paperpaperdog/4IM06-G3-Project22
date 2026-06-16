#!/usr/bin/env bash
# Resume NPU training from last.pt (preprocess already done).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes"

cd "$CODES"
SKIP_PREPARE=1 \
RESUME="${RESUME:-$ROOT/outputs/v1_final64_poscnn/checkpoints/last.pt}" \
EPOCHS="${EPOCHS:-50}" \
JOB="${JOB:-cnn_spectral_v1_npu_resume}" \
bash vc_cnn_spectral_v1.sh
