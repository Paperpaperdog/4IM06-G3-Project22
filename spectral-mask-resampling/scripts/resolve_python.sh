#!/usr/bin/env bash
# Pick a Python interpreter for the mask pipeline.
#   source scripts/resolve_python.sh cpu   # prepare (needs PIL, numpy, torch)
#   source scripts/resolve_python.sh npu   # train/eval on Ascend (torch_npu)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-cpu}"

# Prepare imports used by preprocess_spectra (tv residual + spectrum).
PREPARE_IMPORTS="import PIL, numpy, torch, skimage, yaml, tqdm"

venv_ok() {
  [[ -x "$1/bin/python" ]] && "$1/bin/python" -c "$PREPARE_IMPORTS" >/dev/null 2>&1
}

resolve_cpu() {
  if [[ -n "${PYTHON:-}" ]] && "$PYTHON" -c "$PREPARE_IMPORTS" >/dev/null 2>&1; then
    echo "Using PYTHON=$PYTHON (cpu/prepare)"
    return 0
  fi
  # spectral-mask-resampling and CNN/spectral-history-cnn are siblings under the repo root.
  local cnn_root="${CNN_ROOT:-$ROOT/../CNN/spectral-history-cnn}"
  local legacy_root="${ROOT/-integration/}"
  local -a _venv_candidates=()
  [[ -n "${VENV_DIR:-}" ]] && _venv_candidates+=("$VENV_DIR")
  _venv_candidates+=(
    "$ROOT/.venv"
    "$legacy_root/.venv"
    "$cnn_root/.venv"
  )
  for v in "${_venv_candidates[@]}"; do
    [[ -n "$v" ]] || continue
    if venv_ok "$v"; then
      export PYTHON="$v/bin/python"
      # shellcheck disable=SC1091
      source "$v/bin/activate"
      echo "Using venv for prepare: $v"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1 && python3 -c "$PREPARE_IMPORTS" >/dev/null 2>&1; then
    export PYTHON="python3"
    echo "Using system python3 (cpu/prepare)"
    return 0
  fi
  echo "ERROR: no Python with prepare deps (PIL, numpy, torch, skimage, yaml, tqdm)." >&2
  echo "  Expected venv at one of:" >&2
  echo "    $ROOT/.venv" >&2
  echo "    $legacy_root/.venv" >&2
  echo "  Or set VENV_DIR=/path/to/.venv and retry." >&2
  echo "  Create new venv: cd $ROOT && bash scripts/setup_cpu_venv.sh" >&2
  return 1
}

resolve_npu() {
  # Leave the CPU venv used for prepare — it shadows CANN/tbe on NPU nodes.
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate 2>/dev/null || true
  fi
  unset PYTHONPATH

  export PYTHONNOUSERSITE=1
  export DEVICE=npu

  local cnn_root="${CNN_ROOT:-$ROOT/../CNN/spectral-history-cnn}"
  local activate="$cnn_root/scripts/activate_python.sh"
  if [[ ! -f "$activate" ]]; then
    echo "ERROR: CNN activate_python.sh not found: $activate" >&2
    echo "  Set CNN_ROOT to .../CNN/spectral-history-cnn and retry." >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$activate"

  local py="${PYTHON:-python3}"
  # Mask src/ imports still need the project root on PYTHONPATH.
  export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"

  if ! PYTHONNOUSERSITE=1 "$py" -c "import tbe" >/dev/null 2>&1; then
    echo "WARN: tbe not on PYTHONPATH; re-sourcing Ascend set_env.sh" >&2
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
      # shellcheck disable=SC1091
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"
    fi
  fi

  if PYTHONNOUSERSITE=1 "$py" -c "import tbe; import torch_npu; import torch; assert torch.npu.is_available()" >/dev/null 2>&1; then
    export PYTHON="$py"
    echo "Using NPU python (CANN+tbe ready): $py"
    return 0
  fi

  echo "ERROR: NPU python missing tbe/torch_npu. Source Ascend env on the compute node." >&2
  PYTHONNOUSERSITE=1 "$py" -c "import sys; print('PYTHONPATH', sys.path[:5])" 2>&1 || true
  return 1
}

case "$MODE" in
  cpu) resolve_cpu ;;
  npu) resolve_npu ;;
  *) echo "Usage: source scripts/resolve_python.sh [cpu|npu]" >&2; return 1 2>/dev/null || exit 1 ;;
esac
