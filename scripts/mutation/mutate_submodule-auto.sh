#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HCCL_DETERMINISTIC="${HCCL_DETERMINISTIC:-true}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"
export NCCL_DETERMINISTIC="${NCCL_DETERMINISTIC:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

NPUS_PER_NODE="${NPUS_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
SEQ_LEN="${SEQ_LEN:-1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/assets/runtime/tokenizers/baichuan2}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${REPO_ROOT}/assets/runtime/model_config}"
MUTATE_ENTRY="${MUTATE_ENTRY:-${REPO_ROOT}/utils/runtime/mutate_and_forward/mutate_submodule-auto.py}"

DEFAULT_MUTATE_ARGS="-c ${MODEL_CONFIG_DIR} -r 10 --mutnm 2 -n 3 -m ${MODEL_CONFIG_DIR}/glm4.yaml,${MODEL_CONFIG_DIR}/glm4.yaml,${MODEL_CONFIG_DIR}/glm4.yaml --sub 4,3,5"
MUTATE_ARGS="${MUTATE_ARGS:-${DEFAULT_MUTATE_ARGS}}"

distributed_args=(
    --nproc_per_node "${NPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

gpt_args=(
    --num-layers 16
    --hidden-size 928
    --ffn-hidden-size 1712
    --num-attention-heads 8
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path "${TOKENIZER_PATH}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --micro-batch-size 1
    --global-batch-size 8
)

# shellcheck disable=SC2086
torchrun "${distributed_args[@]}" "${MUTATE_ENTRY}" "${gpt_args[@]}" ${MUTATE_ARGS}
