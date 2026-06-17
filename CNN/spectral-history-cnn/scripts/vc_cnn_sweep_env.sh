#!/usr/bin/env bash
# Reference: add these exports to your cluster-side vc_cnn_spectral_v1.sh when
# submitting from 4IM06-G3-Project22-integration (size_sweep configs live there).
#
# Copy to $CODES or merge into vc_cnn_spectral_v1.sh before the worker exec line.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Codes/4IM06-G3-Project22-integration}"
CNN_ROOT="${CNN_ROOT:-$REPO_ROOT/CNN/spectral-history-cnn}"
WORKER="$CNN_ROOT/scripts/run_v1_gpu.sh"
RAISE_DIR="${RAISE_DIR:-$REPO_ROOT/spectral-mask-resampling/data/raw/raise_tiff}"
# TIFF cache may still be in the non-integration checkout:
if [[ ! -d "$RAISE_DIR" && -d "${REPO_ROOT/-integration/}/spectral-mask-resampling/data/raw/raise_tiff" ]]; then
  RAISE_DIR="${REPO_ROOT/-integration/}/spectral-mask-resampling/data/raw/raise_tiff"
fi
VENV_DIR="${VENV_DIR:-${REPO_ROOT/-integration/}/spectral-mask-resampling/.venv}"

export CNN_ROOT RAISE_DIR VENV_DIR CONFIG="${CONFIG:-configs/size_sweep/n6_poscnn_size64.yaml}"
export SKIP_PREPARE="${SKIP_PREPARE:-0}" EPOCHS="${EPOCHS:-50}" WORKERS="${WORKERS:-18}"
export JOB="${JOB:-n6_cnn}"

if [[ ! -f "$WORKER" ]]; then
  echo "ERROR: worker not found: $WORKER" >&2
  exit 1
fi
if [[ ! -f "$CNN_ROOT/$CONFIG" ]]; then
  echo "ERROR: config not found: $CNN_ROOT/$CONFIG" >&2
  exit 1
fi

echo "vc_cnn sweep: JOB=$JOB CNN_ROOT=$CNN_ROOT CONFIG=$CONFIG RAISE_DIR=$RAISE_DIR VENV_DIR=$VENV_DIR"
# Replace with your real vc submit line (copy from existing vc_cnn_spectral_v1.sh):
#   vc submit ... --cmd "CNN_ROOT=$CNN_ROOT RAISE_DIR=$RAISE_DIR VENV_DIR=$VENV_DIR CONFIG=$CONFIG ... bash $WORKER"
bash "$WORKER"
