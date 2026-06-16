#!/usr/bin/env bash
# Vendor CNN deps for container torch_npu: copy from project venv (no pip on compute node).
#
# Env overrides:
#   NPU_VENV_SOURCE  - venv to copy from (default: .venv, then spectral-mask-resampling/.venv)
#   NPU_VENDOR_DIR   - target dir (default: .pydeps_npu)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3}"
VENDOR="${NPU_VENDOR_DIR:-$ROOT/.pydeps_npu}"

REQUIRED_IMPORTS=(matplotlib pandas yaml PIL tqdm sklearn)

export_vendor_path() {
  if [[ -d "$VENDOR" ]]; then
    export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

import_ok() {
  PYTHONNOUSERSITE=1 PYTHONPATH="${VENDOR}${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -c "import $1" >/dev/null 2>&1
}

all_present() {
  local mod
  for mod in "${REQUIRED_IMPORTS[@]}"; do
    if ! import_ok "$mod"; then
      return 1
    fi
  done
  return 0
}

resolve_venv() {
  local candidate
  for candidate in \
    "${NPU_VENV_SOURCE:-}" \
    "$ROOT/.venv" \
    "$ROOT/../../spectral-mask-resampling/.venv"
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

rsync_excludes=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'torch/'
  --exclude 'torch-*.dist-info/'
  --exclude 'torchgen/'
  --exclude 'torchvision/'
  --exclude 'torchvision-*.dist-info/'
  --exclude 'torchaudio/'
  --exclude 'torchaudio-*.dist-info/'
  --exclude 'torch_npu/'
  --exclude 'torch_npu-*.dist-info/'
  --exclude 'triton/'
  --exclude 'triton-*.dist-info/'
  --exclude 'nvidia_*/'
  --exclude 'pip/'
  --exclude 'pip-*.dist-info/'
  --exclude 'setuptools/'
  --exclude 'setuptools-*.dist-info/'
  --exclude 'wheel/'
  --exclude 'wheel-*.dist-info/'
  --exclude '_distutils_hack/'
  --exclude '_virtualenv*'
)

should_skip_vendor_item() {
  local base="$1"
  case "$base" in
    __pycache__|_distutils_hack|_virtualenv*|pip|pip-*|setuptools|setuptools-*|wheel|wheel-*)
      return 0
      ;;
    torch|torch-*|torchgen|torchvision|torchvision-*|torchaudio|torchaudio-*|torch_npu|torch_npu-*|triton|triton-*)
      return 0
      ;;
    nvidia_*)
      return 0
      ;;
  esac
  return 1
}

copy_tree_fallback() {
  local src="$1" name
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    base="$(basename "$name")"
    should_skip_vendor_item "$base" && continue
    cp -a "$name" "$VENDOR/"
  done
}

copy_from_venv() {
  local venv src
  venv="$(resolve_venv)" || {
    echo "ERROR: no source venv found for NPU deps." >&2
    echo "  Create one: cd $ROOT && python3 -m venv .venv && pip install -r requirements.txt" >&2
    echo "  Or set NPU_VENV_SOURCE=/path/to/venv" >&2
    return 1
  }

  src="$(venv_site_packages "$venv")"
  [[ -d "$src" ]] || {
    echo "ERROR: site-packages missing in venv: $src" >&2
    return 1
  }

  mkdir -p "$VENDOR"
  echo "Copying non-torch deps from $venv"
  echo "  site-packages: $src"
  echo "  vendor dir:    $VENDOR"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${rsync_excludes[@]}" "$src/" "$VENDOR/"
  else
    echo "WARN: rsync not found, using cp -a" >&2
    copy_tree_fallback "$src"
  fi

  venv_py="$("$venv/bin/python" -V 2>&1)"
  container_py="$(PYTHONNOUSERSITE=1 "$PY" -V 2>&1 || true)"
  {
    echo "source_venv=$venv"
    echo "source_python=$venv_py"
    echo "container_python=$container_py"
    echo "copied_at=$(date -Is 2>/dev/null || date)"
  } > "$VENDOR/.vendor_stamp"
}

export_vendor_path

if all_present; then
  echo "NPU python deps OK in $VENDOR (skip copy)"
  return 0 2>/dev/null || exit 0
fi

copy_from_venv
export_vendor_path

if ! all_present; then
  echo "ERROR: deps copied but imports still fail on container python." >&2
  echo "  Ensure venv Python matches container (e.g. 3.10) and requirements are installed:" >&2
  echo "    cd $ROOT && source .venv/bin/activate && pip install -r requirements.txt" >&2
  echo "    bash scripts/ensure_npu_deps.sh python3" >&2
  PYTHONNOUSERSITE=1 PYTHONPATH="$VENDOR" "$PY" - <<'PY' 2>&1 || true
mods = ["matplotlib", "pandas", "yaml", "PIL", "tqdm", "sklearn"]
for m in mods:
    try:
        __import__(m)
        print("ok", m)
    except Exception as e:
        print("fail", m, e)
PY
  exit 1
fi

echo "NPU python deps ready in $VENDOR"
