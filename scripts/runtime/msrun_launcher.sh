#!/usr/bin/env bash
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -euo pipefail

WORKER_NUM=8
LOCAL_WORKER=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=8118
NODE_RANK=${NODE_RANK:-0}
LOG_DIR="msrun_log"
JOIN="False"
CLUSTER_TIME_OUT=7200
SINGLE_NODE=false

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_PATH="${REPO_ROOT}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MF_LOG_SUFFIX="${MF_LOG_SUFFIX:-}"
if [ -n "${MF_LOG_SUFFIX}" ]; then
  MF_LOG_SUFFIX="_${MF_LOG_SUFFIX}"
fi

if [ -z "${LOG_MF_PATH:-}" ]; then
  export LOG_MF_PATH="${WORKSPACE_PATH}/output/log${MF_LOG_SUFFIX}"
fi

if [ -n "${LMSV_MSRUN_MASTER_PORT:-}" ]; then
  MASTER_PORT="${LMSV_MSRUN_MASTER_PORT}"
fi

if [ "${PLOG_REDIRECT_TO_OUTPUT:-False}" = "False" ]; then
  echo "No change the path of plog, the path of plog is /root/ascend"
else
  export ASCEND_PROCESS_LOG_PATH="${WORKSPACE_PATH}/output/plog${MF_LOG_SUFFIX}"
  echo "PLOG_REDIRECT_TO_OUTPUT=${PLOG_REDIRECT_TO_OUTPUT}, set the path of plog to ${ASCEND_PROCESS_LOG_PATH}"
fi

usage() {
  echo "Usage Help: bash msrun_launcher.sh [EXECUTE_ORDER] For Default 8 Devices In Single Machine"
  echo "Usage Help: bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] For Quick Start On Multiple Devices In Single Machine"
  echo "Usage Help: bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [MASTER_PORT] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT] For Multiple Devices In Single Machine"
  echo "Usage Help: bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [LOCAL_WORKER] [MASTER_ADDR] [MASTER_PORT] [NODE_RANK] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT] For Multiple Devices In Multiple Machines"
  exit 1
}

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ] && [ "$#" -ne 6 ] && [ "$#" -ne 9 ]; then
  usage
fi

EXECUTE_ORDER_RAW="$1"

if [ "$#" -eq 1 ]; then
  echo "No parameter is entered. Notice that the program will run on default 8 cards."
  SINGLE_NODE=true
else
  WORKER_NUM="$2"
fi

if [[ ! "${WORKER_NUM}" =~ ^[0-9]+$ ]]; then
  echo "error: worker_num=${WORKER_NUM} is not a number"
  exit 1
fi

if [ "$#" -eq 2 ]; then
  LOCAL_WORKER="${WORKER_NUM}"
  SINGLE_NODE=true
fi

if [ "$#" -eq 6 ]; then
  LOCAL_WORKER="${WORKER_NUM}"
  MASTER_PORT="$3"
  LOG_DIR="$4"
  JOIN="$5"
  CLUSTER_TIME_OUT="$6"
  SINGLE_NODE=true
fi

if [ "$#" -eq 9 ]; then
  LOCAL_WORKER="$3"
  MASTER_ADDR="$4"
  MASTER_PORT="$5"
  NODE_RANK="$6"
  LOG_DIR="$7"
  JOIN="$8"
  CLUSTER_TIME_OUT="$9"

  if [ "${WORKER_NUM}" -eq "${LOCAL_WORKER}" ]; then
    echo "worker_num is equal to local_worker, Notice that task will run on single node."
    SINGLE_NODE=true
  else
    echo "worker_num=${WORKER_NUM}, local_worker=${LOCAL_WORKER}, Please run this script on other nodes with different node_rank."
    SINGLE_NODE=false
  fi
fi

LOG_DIR="${LOG_DIR}${MF_LOG_SUFFIX}"

if [ "${WORKER_NUM}" -eq 1 ]; then
  echo "You should use python instead of using msrun while running a single rank"
  exit 0
fi

if [ "${SINGLE_NODE}" = true ]; then
  MSRUN_CMD="msrun --bind_core=True \
   --worker_num=${WORKER_NUM} \
   --local_worker_num=${LOCAL_WORKER} \
   --master_port=${MASTER_PORT} \
   --log_dir=${LOG_DIR} \
   --join=${JOIN} \
   --cluster_time_out=${CLUSTER_TIME_OUT}"
else
  MSRUN_CMD="msrun --bind_core=True \
   --worker_num=${WORKER_NUM} \
   --local_worker_num=${LOCAL_WORKER} \
   --master_addr=${MASTER_ADDR} \
   --master_port=${MASTER_PORT} \
   --node_rank=${NODE_RANK} \
   --log_dir=${LOG_DIR} \
   --join=${JOIN} \
   --cluster_time_out=${CLUSTER_TIME_OUT}"
fi

EXECUTE_ORDER="${MSRUN_CMD} ${EXECUTE_ORDER_RAW}"

current_hard_limit=$(ulimit -H -u)
ulimit -u $current_hard_limit

echo "Running Command: ${EXECUTE_ORDER}"
echo "Please check log files in ${WORKSPACE_PATH}/${LOG_DIR}"

eval "${EXECUTE_ORDER}"
