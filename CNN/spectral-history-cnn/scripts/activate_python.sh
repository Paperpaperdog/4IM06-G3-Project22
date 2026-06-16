#!/usr/bin/env bash
# Pick Python: container torch_npu (NPU train) > project venv (CPU preprocess).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/setup_npu_env.sh"

python_has_npu() {
  # User-site torch (~/.local) conflicts with container torch_npu ABI.
  PYTHONNOUSERSITE=1 "$1" -c "import torch_npu; import torch; assert torch.npu.is_available()" >/dev/null 2>&1
}

venv_has_torch() {
  "$1/bin/python" -c "import torch" >/dev/null 2>&1
}

pick_container_npu_python() {
  local py
  for py in "${PYTHON:-}" python3 /usr/local/python3.10.15/bin/python3; do
    [[ -n "$py" ]] || continue
    command -v "$py" >/dev/null 2>&1 || [[ -x "$py" ]] || continue
    if python_has_npu "$py"; then
      export PYTHONNOUSERSITE=1
      if [[ "$py" != python3* ]] && [[ "$py" != /usr/bin/python3 ]]; then
        export PYTHON="$py"
      fi
      echo "Using container python (torch_npu): $py"
      return 0
    fi
  done
  return 1
}

if [[ "${DEVICE:-}" == "npu" ]]; then
  if pick_container_npu_python; then
  PY="${PYTHON:-python3}"
  # shellcheck disable=SC1091
  source "$ROOT/scripts/ensure_npu_deps.sh" "$PY"
  else
    echo "ERROR: DEVICE=npu but no working torch_npu python found." >&2
    echo "  Ensure Ascend env is loaded and ~/.local torch is not shadowing container torch." >&2
    PYTHONNOUSERSITE=1 python3 -c "import torch; print('torch', torch.__file__)" 2>&1 || true
    exit 1
  fi
elif [[ -n "${PYTHON:-}" ]] && python_has_npu "$PYTHON"; then
  export PYTHONNOUSERSITE=1
  echo "Using PYTHON=$PYTHON (torch_npu)"
elif python_has_npu python3; then
  export PYTHONNOUSERSITE=1
  echo "Using container python3 (torch_npu)"
elif [[ -d "$ROOT/.venv" ]] && venv_has_torch "$ROOT/.venv"; then
  echo "Using venv: $ROOT/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -d "$ROOT/../../spectral-mask-resampling/.venv" ]] \
  && venv_has_torch "$ROOT/../../spectral-mask-resampling/.venv"; then
  echo "Using fallback venv: spectral-mask-resampling/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/../../spectral-mask-resampling/.venv/bin/activate"
else
  echo "Using system python3"
fi
