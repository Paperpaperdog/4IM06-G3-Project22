#!/usr/bin/env bash
# Install CNN deps into a local target dir for container torch_npu python (no torch reinstall).
#
# Env overrides:
#   NPU_PIP_INDEX_URL  - PyPI mirror (falls back to PIP_INDEX_URL), e.g. https://pypi.tuna.tsinghua.edu.cn/simple
#   NPU_PIP_WHEELS     - offline wheel dir (default: wheels_npu/); used when *.whl present
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3}"
VENDOR="${ROOT}/.pydeps_npu"
WHEELS="${NPU_PIP_WHEELS:-$ROOT/wheels_npu}"

export_vendor_path() {
  if [[ -d "$VENDOR" ]]; then
    export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

import_ok() {
  PYTHONNOUSERSITE=1 PYTHONPATH="${VENDOR}${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -c "import $1" >/dev/null 2>&1
}

# pip package name -> python import name
REQUIRED=(
  matplotlib:matplotlib
  pandas:pandas
  pyyaml:yaml
  pillow:PIL
  tqdm:tqdm
  scikit-learn:sklearn
)

all_present() {
  local pair pkg mod
  for pair in "${REQUIRED[@]}"; do
    mod="${pair#*:}"
    if ! import_ok "$mod"; then
      return 1
    fi
  done
  return 0
}

export_vendor_path

if all_present; then
  echo "NPU python deps OK in $VENDOR (skip pip install)"
  return 0 2>/dev/null || exit 0
fi

mkdir -p "$VENDOR"

pip_base_args=(
  -m pip install -q --upgrade
  --target "$VENDOR"
)

index_url="${NPU_PIP_INDEX_URL:-${PIP_INDEX_URL:-}}"
if [[ -n "$index_url" ]]; then
  pip_base_args+=(--index-url "$index_url")
  index_host="${index_url#*://}"
  index_host="${index_host%%/*}"
  index_host="${index_host%%:*}"
  [[ -n "$index_host" ]] && pip_base_args+=(--trusted-host "$index_host")
fi

# HPC clusters often fail PyPI HTTPS with weak CA / EE cert errors.
pip_base_args+=(
  --trusted-host pypi.org
  --trusted-host pypi.python.org
  --trusted-host files.pythonhosted.org
)

PRIMARY_PKGS=(matplotlib pandas pyyaml pillow tqdm scikit-learn)
EXTRA_PKGS=(numpy scipy contourpy cycler fonttools kiwisolver packaging pyparsing python-dateutil pytz tzdata six)

pip_install() {
  local extra_args=("$@")
  PYTHONNOUSERSITE=1 "$PY" "${pip_base_args[@]}" "${extra_args[@]}"
}

has_offline_wheels() {
  [[ -d "$WHEELS" ]] && compgen -G "$WHEELS/*.whl" >/dev/null
}

echo "Installing CNN python deps into $VENDOR (no torch)..."

if has_offline_wheels; then
  echo "Using offline wheels from $WHEELS"
  pip_install --no-index --find-links "$WHEELS" --no-deps "${PRIMARY_PKGS[@]}"
  pip_install --no-index --find-links "$WHEELS" "${EXTRA_PKGS[@]}"
else
  pip_install --no-deps "${PRIMARY_PKGS[@]}"
  pip_install "${EXTRA_PKGS[@]}"
fi

export_vendor_path

if ! all_present; then
  echo "ERROR: NPU deps install finished but imports still fail." >&2
  echo "  Try on login node:" >&2
  echo "    NPU_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/ensure_npu_deps.sh python3" >&2
  echo "  Or place wheels in $WHEELS and re-run." >&2
  exit 1
fi

echo "NPU python deps ready in $VENDOR"
