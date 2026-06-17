#!/usr/bin/env bash
# Submit mask spectral-mask-resampling pipeline to Ascend NPU via vc (pdgpu-sjtu-ai).
# Copy to ~/Codes/vc_mask.sh next to vc_cnn_spectral_v1.sh.
#
# From integration repo:
#   cd 4IM06-G3-Project22-integration/spectral-mask-resampling
#   SKIP_PREPARE=1 bash scripts/submit_size_sweep_npu.sh
#
# Single size:
#   REPO_ROOT=.../4IM06-G3-Project22-integration \
#   CONFIG=configs/size_sweep/n6_mask_size64.yaml \
#   JOB=n6_mask_size64 SKIP_PREPARE=1 \
#   bash ~/Codes/vc_mask.sh
set -euo pipefail

n_gpus="${N_GPUS:-1}"
PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
IMAGE="${IMAGE:-hub.szaic.com/sjtu/sjtu_wumengyue-kunyao.lan:vllm_25.4.12}"
JOB="${JOB:-n6_mask}"

export REPO_ROOT="${REPO_ROOT:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22-integration}"
export CNN_ROOT="${CNN_ROOT:-$REPO_ROOT/CNN/spectral-history-cnn}"
export VENV_DIR="${VENV_DIR:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22/spectral-mask-resampling/.venv}"

MASK_ROOT="$REPO_ROOT/spectral-mask-resampling"
WORKER="${MASK_ROOT}/scripts/vc_worker.sh"
if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: worker not found: $WORKER" >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/size_sweep/n6_mask_size64.yaml}"
if [[ ! -f "${MASK_ROOT}/${CONFIG}" ]]; then
  echo "ERROR: config not found: ${MASK_ROOT}/${CONFIG}" >&2
  exit 1
fi

CMD="REPO_ROOT=${REPO_ROOT} CNN_ROOT=${CNN_ROOT} VENV_DIR=${VENV_DIR} CONFIG=${CONFIG} bash ${WORKER}"

[[ -n "${SKIP_PREPARE:-}" ]] && CMD="SKIP_PREPARE=${SKIP_PREPARE} ${CMD}"
[[ -n "${EVAL_ONLY:-}"   ]] && CMD="EVAL_ONLY=${EVAL_ONLY} ${CMD}"

echo "Submitting JOB=$JOB"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  MASK_ROOT=$MASK_ROOT"
echo "  CONFIG=$CONFIG"
echo "  WORKER=$WORKER"

set -x
vc submit \
    --image "$IMAGE" \
    --partition "$PARTITION" \
    --gpu-per-task "${n_gpus}" \
    --job "$JOB" \
    --mem-per-task "${MEM_PER_TASK:-64G}" \
    --cpu-per-task "${CPU_PER_TASK:-20}" \
    --num-task 1 \
    --cmd "$CMD"
