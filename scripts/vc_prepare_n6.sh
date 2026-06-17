#!/usr/bin/env bash
# Submit CPU-only n6 spectrum prepare via vc. Copy to ~/Codes/vc_prepare_n6.sh.
#
#   REPO_ROOT=.../4IM06-G3-Project22-integration SIZE=64 JOB=n6_prepare_size64 \
#     bash ~/Codes/vc_prepare_n6.sh
set -euo pipefail

PARTITION="${PARTITION:-pdgpu-sjtu-ai}"
IMAGE="${IMAGE:-hub.szaic.com/sjtu/sjtu_wumengyue-kunyao.lan:vllm_25.4.12}"
JOB="${JOB:-n6_prepare_size64}"

export REPO_ROOT="${REPO_ROOT:-/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes/4IM06-G3-Project22-integration}"
SIZE="${SIZE:-64}"
PREP_WORKERS="${PREP_WORKERS:-0}"

WORKER="${REPO_ROOT}/scripts/prepare_n6_worker.sh"
if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: worker not found: $WORKER" >&2
  exit 1
fi

CMD="REPO_ROOT=${REPO_ROOT} SIZE=${SIZE} PREP_WORKERS=${PREP_WORKERS} bash ${WORKER}"

echo "Submitting JOB=$JOB (CPU prepare, no NPU)"
echo "  REPO_ROOT=$REPO_ROOT SIZE=$SIZE WORKER=$WORKER"

set -x
vc submit \
    --image "$IMAGE" \
    --partition "$PARTITION" \
    --gpu-per-task 0 \
    --job "$JOB" \
    --mem-per-task "${MEM_PER_TASK:-32G}" \
    --cpu-per-task "${CPU_PER_TASK:-16}" \
    --num-task 1 \
    --cmd "$CMD"
