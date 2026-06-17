>>>>>>> 7de3649 (supplementary experiment)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODES="/aistor/sjtu/hpc_stor01/home/jinbingrui/Codes"

export JOB="${JOB:-cnn_spectral_v1_npu_resume}"
export CONFIG="${CONFIG:-configs/v1_final64_poscnn_local.yaml}"

if [[ "$JOB" == *full* ]]; then
  export SKIP_PREPARE="${SKIP_PREPARE:-0}"
else
  export SKIP_PREPARE="${SKIP_PREPARE:-1}"
fi

REQUESTED_RESUME="${RESUME-__unset__}"
if [[ "${REQUESTED_RESUME}" == "none" ]]; then
  unset RESUME
elif [[ "${REQUESTED_RESUME}" != "__unset__" && -n "${REQUESTED_RESUME}" ]]; then
  export RESUME="${REQUESTED_RESUME}"
elif [[ "$SKIP_PREPARE" == "1" ]]; then
  OUTPUT_DIR="$(python3 -c "import sys; sys.path.insert(0,'${ROOT}'); from src.utils.io import load_yaml; print(load_yaml('${ROOT}/${CONFIG}')['paths']['output_dir'])")"
  export RESUME="${ROOT}/${OUTPUT_DIR}/checkpoints/last.pt"
fi

if [[ -n "${EPOCHS+x}" ]]; then
  export EPOCHS
fi

cd "$CODES"
<<<<<<< HEAD
SKIP_PREPARE="${SKIP_PREPARE:-1}" \
CONFIG="${CONFIG:-$ROOT/configs/v1_final64_poscnn_local.yaml}" \
RESUME="${RESUME:-}" \
EPOCHS="${EPOCHS:-50}" \
JOB="${JOB:-cnn_spectral_v1_npu_4cls}" \
=======
>>>>>>> 7de3649 (supplementary experiment)
bash vc_cnn_spectral_v1.sh
