#!/usr/bin/env bash
# Pick a Python interpreter for the mask pipeline.
#   source scripts/resolve_python.sh cpu   # prepare (needs PIL, numpy, torch)
#   source scripts/resolve_python.sh npu   # train/eval on Ascend (torch_npu)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-cpu}"

venv_ok() {
  [[ -x "$1/bin/python" ]] && "$1/bin/python" -c "import PIL, numpy, torch" >/dev/null 2>&1
}

resolve_cpu() {
  if [[ -n "${PYTHON:-}" ]] && "$PYTHON" -c "import PIL, numpy, torch" >/dev/null 2>&1; then
    echo "Using PYTHON=$PYTHON (cpu/prepare)"
    return 0
  fi
  for v in "$ROOT/.venv" "$ROOT/../../CNN/spectral-history-cnn/.venv"; do
    if venv_ok "$v"; then
      # shellcheck disable=SC1091
      source "$v/bin/activate"
      echo "Using venv for prepare: $v"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1 && python3 -c "import PIL, numpy, torch" >/dev/null 2>&1; then
    echo "Using system python3 (cpu/prepare)"
    return 0
  fi
  echo "ERROR: no Python with PIL+numpy+torch for prepare." >&2
  echo "  Create a venv and install deps:" >&2
  echo "    cd $ROOT && python3 -m venv .venv && source .venv/bin/activate" >&2
  echo "    pip install -r requirements.txt" >&2
  return 1
}

resolve_npu() {
  export PYTHONNOUSERSITE=1
  NPU_ENV="$ROOT/../../CNN/spectral-history-cnn/scripts/setup_npu_env.sh"
  if [[ -f "$NPU_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$NPU_ENV" || true
  fi
  for py in "${PYTHON:-}" python3 /usr/local/python3.10.15/bin/python3; do
    [[ -n "$py" ]] || continue
    command -v "$py" >/dev/null 2>&1 || [[ -x "$py" ]] || continue
    if PYTHONNOUSERSITE=1 "$py" -c "import torch_npu; import torch; assert torch.npu.is_available()" >/dev/null 2>&1; then
      export PYTHON="$py"
      echo "Using NPU python: $py"
      ENSURE="$ROOT/../../CNN/spectral-history-cnn/scripts/ensure_npu_deps.sh"
      if [[ -f "$ENSURE" ]]; then
        # shellcheck disable=SC1091
        source "$ENSURE" "$py" || true
      fi
      return 0
    fi
  done
  echo "WARN: torch_npu not found; falling back to CPU venv (train will fail on NPU config)." >&2
  resolve_cpu
}

case "$MODE" in
  cpu) resolve_cpu ;;
  npu) resolve_npu ;;
  *) echo "Usage: source scripts/resolve_python.sh [cpu|npu]" >&2; return 1 2>/dev/null || exit 1 ;;
esac
