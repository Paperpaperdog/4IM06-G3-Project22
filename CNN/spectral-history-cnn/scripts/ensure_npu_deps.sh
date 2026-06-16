#!/usr/bin/env bash
# Install CNN deps into a local target dir for container torch_npu python (no torch reinstall).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3}"
VENDOR="${ROOT}/.pydeps_npu"

missing() {
  ! PYTHONNOUSERSITE=1 "$PY" -c "import $1" >/dev/null 2>&1
}

if ! missing matplotlib && ! missing pandas && ! missing yaml; then
  if [[ -d "$VENDOR" ]]; then
    export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
  fi
  return 0 2>/dev/null || exit 0
fi

mkdir -p "$VENDOR"
echo "Installing CNN python deps into $VENDOR (no torch)..."
PYTHONNOUSERSITE=1 "$PY" -m pip install -q --upgrade \
  --target "$VENDOR" \
  --no-deps \
  matplotlib pandas pyyaml pillow tqdm scikit-learn

PYTHONNOUSERSITE=1 "$PY" -m pip install -q --upgrade \
  --target "$VENDOR" \
  contourpy cycler fonttools kiwisolver packaging pyparsing python-dateutil pytz tzdata six

export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
