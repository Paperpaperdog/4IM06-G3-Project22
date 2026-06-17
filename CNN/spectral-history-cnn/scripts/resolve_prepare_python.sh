#!/usr/bin/env bash
# Pick a CPU Python with torch+skimage for CNN/Mask preprocess on NPU nodes.
# Usage: source scripts/resolve_prepare_python.sh
set -euo pipefail

_CNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CNN_ROOT="${CNN_ROOT:-$_CNN_ROOT}"
repo_root="$(cd "$CNN_ROOT/../.." && pwd)"
legacy_repo="${repo_root/-integration/}"

PREPARE_IMPORTS="import torch, skimage, yaml, tqdm"

prep_ok() {
  [[ -x "$1" ]] && "$1" -c "$PREPARE_IMPORTS" >/dev/null 2>&1
}

resolve_prepare_python() {
  local candidate
  local -a candidates=()
  [[ -n "${VENV_DIR:-}" ]] && candidates+=("$VENV_DIR/bin/python")
  candidates+=(
    "$CNN_ROOT/.venv/bin/python"
    "$repo_root/spectral-mask-resampling/.venv/bin/python"
    "$legacy_repo/spectral-mask-resampling/.venv/bin/python"
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    if prep_ok "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if PREP_PY="$(resolve_prepare_python)"; then
  export PREP_PY
  echo "Using prepare python: $PREP_PY"
else
  echo "ERROR: no Python with torch+skimage for prepare." >&2
  echo "  Set VENV_DIR=/path/to/mask/.venv and retry." >&2
  echo "  Tried: VENV_DIR=${VENV_DIR:-<unset>}" >&2
  echo "    $legacy_repo/spectral-mask-resampling/.venv" >&2
  return 1 2>/dev/null || exit 1
fi
