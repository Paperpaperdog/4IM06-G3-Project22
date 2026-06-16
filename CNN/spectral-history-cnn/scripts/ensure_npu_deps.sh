#!/usr/bin/env bash
# Minimal pure-Python deps for container torch_npu (never vendor torch/numpy/pandas/matplotlib).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3}"
VENDOR="${NPU_VENDOR_DIR:-$ROOT/.pydeps_npu}"

export_vendor_path() {
  if [[ -d "$VENDOR" ]] && { [[ -d "$VENDOR/yaml" ]] || [[ -d "$VENDOR/_yaml" ]]; }; then
    export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

yaml_ready() {
  PYTHONNOUSERSITE=1 PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -c "import yaml" >/dev/null 2>&1
}

resolve_venv() {
  local candidate
  for candidate in \
    "${NPU_VENV_SOURCE:-}" \
    "$ROOT/../../spectral-mask-resampling/.venv" \
    "$ROOT/.venv"
  do
    [[ -n "$candidate" && -x "$candidate/bin/python" ]] || continue
    echo "$candidate"
    return 0
  done
  return 1
}

venv_site_packages() {
  local venv="$1"
  "$venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
}

copy_pure_python_deps() {
  local venv src item
  venv="$(resolve_venv)" || {
    echo "ERROR: no source venv for PyYAML." >&2
    return 1
  }
  src="$(venv_site_packages "$venv")"
  mkdir -p "$VENDOR"
  echo "Copying PyYAML from $venv -> $VENDOR"
  for item in "$src"/yaml "$src"/_yaml "$src"/PyYAML-*.dist-info; do
    [[ -e "$item" ]] || continue
    cp -a "$item" "$VENDOR/"
  done
}

strip_heavy_vendor() {
  [[ -d "$VENDOR" ]] || return 0
  local item base
  for item in "$VENDOR"/*; do
    [[ -e "$item" ]] || continue
    base="$(basename "$item")"
    case "$base" in
      yaml|_yaml|PyYAML-*|.vendor_stamp)
        ;;
      *)
        rm -rf "$item"
        ;;
    esac
  done
}

strip_heavy_vendor
copy_pure_python_deps
export_vendor_path

if yaml_ready; then
  echo "NPU python deps ready in $VENDOR (PyYAML only)"
  return 0 2>/dev/null || exit 0
fi

echo "ERROR: PyYAML not available for NPU training." >&2
exit 1
