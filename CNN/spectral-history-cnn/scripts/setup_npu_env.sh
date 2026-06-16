#!/usr/bin/env bash
# Source Ascend toolkit env (for vc NPU containers).
# Ascend's set_env.sh references unset vars (e.g. ZSH_VERSION); disable nounset while sourcing.
_npu_restore_u=0
[[ $- == *u* ]] && _npu_restore_u=1 && set +u

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
# ATB env is for vLLM inference; optional for torch_npu CNN training.
if [[ "${ASCEND_SOURCE_ATB:-0}" == "1" ]] && [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

[[ $_npu_restore_u -eq 1 ]] && set -u
unset _npu_restore_u

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
