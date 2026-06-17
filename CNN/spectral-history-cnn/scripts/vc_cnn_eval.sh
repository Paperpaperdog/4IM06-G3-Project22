#!/usr/bin/env bash
# Optional: submit CNN test eval via vc (CPU, gpu-per-task 0).
# Prefer direct run on debug/login node instead:
#   bash scripts/run_cnn_eval_all.sh
#
# Copy to ~/Codes/vc_cnn_eval.sh only if you need detached vc jobs.
set -euo pipefail

PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
IMAGE="${IMAGE:-hub.szaic.com/sjtu/sjtu_wumengyue-kunyao.lan:vllm_25.4.12}"
JOB="${JOB:-n6_cnn_eval}"

export REPO_ROOT="${REPO_ROOT:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22-integration}"
export CNN_ROOT="${CNN_ROOT:-$REPO_ROOT/CNN/spectral-history-cnn}"
export VENV_DIR="${VENV_DIR:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22/spectral-mask-resampling/.venv}"
CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"

WORKER="${CNN_ROOT}/scripts/run_v1_eval_cpu.sh"
if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: worker not found: $WORKER" >&2
  exit 1
fi
if [[ ! -f "${CNN_ROOT}/${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CNN_ROOT}/${CONFIG}" >&2
  exit 1
fi

CMD="CNN_ROOT=${CNN_ROOT} VENV_DIR=${VENV_DIR} CONFIG=${CONFIG} DEVICE=cpu bash ${WORKER}"

echo "Submitting JOB=$JOB (CPU eval-only, no NPU)"
echo "  CNN_ROOT=$CNN_ROOT CONFIG=$CONFIG"

set -x
vc submit \
    --image "$IMAGE" \
    --partition "$PARTITION" \
    --gpu-per-task 0 \
    --job "$JOB" \
    --mem-per-task "${MEM_PER_TASK:-32G}" \
    --cpu-per-task "${CPU_PER_TASK:-8}" \
    --num-task 1 \
    --cmd "$CMD"
