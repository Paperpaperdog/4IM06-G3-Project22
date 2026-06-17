#!/usr/bin/env bash
# One-time CPU venv for mask prepare (PIL, skimage, torch, etc.).
# Run on a login or CPU node before the first pipeline / size sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
VENV="${VENV_DIR:-$ROOT/.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel
pip install -r requirements.txt

python -c "import PIL, numpy, torch, skimage, yaml, tqdm; print('prepare deps OK')"
echo "CPU venv ready: source $VENV/bin/activate"
