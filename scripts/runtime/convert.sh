#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  # Keep aligned with legacy convert.sh runtime env.
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -n "${PTAPATH:-}" && -f "scripts/envset/pta.sh" ]]; then
  # Formal mindspeed activation requires both conda activation and envset PYTHONPATH setup.
  export ORIGIN_PYTHONPATH="${ORIGIN_PYTHONPATH:-${PYTHONPATH:-}}"
  source scripts/envset/pta.sh
fi

CONVERT_ENTRY="${LMSV_CONVERT_CKPT_ENTRY:-${REPO_ROOT}/utils/runtime/convert_ckpt.py}"

if [[ ! -f "${CONVERT_ENTRY}" ]]; then
  echo "ERROR: convert_ckpt.py not found: ${CONVERT_ENTRY}" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runtime/convert.sh [args passed to convert_ckpt.py]

Or use env injection (when no CLI args are provided):
  LMSV_CONVERT_LOAD_DIR
  LMSV_CONVERT_SAVE_DIR
  LMSV_CONVERT_MODEL_TYPE_HF (default: qwen3)
  LMSV_CONVERT_LOAD_MODEL_TYPE (default: mg)
  LMSV_CONVERT_SAVE_MODEL_TYPE (default: hf)
  LMSV_CONVERT_TARGET_TP (default: 1)
  LMSV_CONVERT_TARGET_PP (default: 1)
  LMSV_CONVERT_TARGET_EP (default: 1)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  python "${CONVERT_ENTRY}" --help
  exit 0
fi

declare -a CONVERT_ARGS
if [[ $# -gt 0 ]]; then
  CONVERT_ARGS=("$@")
else
  LOAD_DIR="${LMSV_CONVERT_LOAD_DIR:-}"
  SAVE_DIR="${LMSV_CONVERT_SAVE_DIR:-}"
  MODEL_TYPE_HF="${LMSV_CONVERT_MODEL_TYPE_HF:-qwen3}"
  LOAD_MODEL_TYPE="${LMSV_CONVERT_LOAD_MODEL_TYPE:-mg}"
  SAVE_MODEL_TYPE="${LMSV_CONVERT_SAVE_MODEL_TYPE:-hf}"
  TARGET_TP="${LMSV_CONVERT_TARGET_TP:-1}"
  TARGET_PP="${LMSV_CONVERT_TARGET_PP:-1}"
  TARGET_EP="${LMSV_CONVERT_TARGET_EP:-1}"

  if [[ -z "${LOAD_DIR}" || -z "${SAVE_DIR}" ]]; then
    echo "ERROR: missing load/save dir. Provide CLI args or set LMSV_CONVERT_LOAD_DIR/LMSV_CONVERT_SAVE_DIR." >&2
    usage
    exit 2
  fi

  CONVERT_ARGS=(
    --load-model-type "${LOAD_MODEL_TYPE}"
    --save-model-type "${SAVE_MODEL_TYPE}"
    --load-dir "${LOAD_DIR}"
    --save-dir "${SAVE_DIR}"
    --model-type-hf "${MODEL_TYPE_HF}"
    --target-tensor-parallel-size "${TARGET_TP}"
    --target-pipeline-parallel-size "${TARGET_PP}"
    --target-expert-parallel-size "${TARGET_EP}"
  )
fi

python "${CONVERT_ENTRY}" "${CONVERT_ARGS[@]}"
