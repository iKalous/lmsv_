#!/bin/bash
# Task6 PTA真实执行脚本 - OpenSora (推理模式，不需要data_config)
set -e

# 设置输出路径（未设置时使用默认值）
export LMSV_OUTPATH="${LMSV_OUTPATH:-output}"

# 设置环境名称（未设置时使用默认值）
export PTA_NAME="${PTA_NAME:-mindspeed}"
export MSA_NAME="${MSA_NAME:-msadapter}"

echo "========================================"
echo "Task6 PTA Real Execution - OpenSora (Inference)"
echo "MM_MODEL: $MM_MODEL"
echo "========================================"

# 清理端口占用（防止上一次执行未正常结束）
echo "Cleaning up ports..."
fuser -k 6000/tcp 2>/dev/null || true
fuser -k 6001/tcp 2>/dev/null || true
fuser -k 6002/tcp 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

if [ -z "$MM_MODEL" ]; then
    echo "ERROR: MM_MODEL not set"
    exit 1
fi

if [ ! -f "$MM_MODEL" ]; then
    echo "ERROR: Model config not found: $MM_MODEL"
    exit 1
fi

echo "Configuration check PASSED"
echo "Using mutated config: $MM_MODEL"
echo "Note: OpenSora is inference mode, no data_config needed"

# 检查数据集根目录环境变量（必须设置，不可硬编码）
if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT environment variable is not set"
    echo "Please set it in config.json or export DATASET_ROOT=/path/to/dataset"
    exit 1
fi
# 注：OpenSora是推理模型，不需要LOAD_PATH

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Task6开始时已清空日志，这里只确保目录存在
LOG_DIR="${LMSV_ROOT}/pta_logs"
mkdir -p ${LOG_DIR}
echo "Using log directory: ${LOG_DIR}"

# 处理配置文件 - 使用配置预处理脚本（替换{{LOAD_PATH}}等占位符）
mkdir -p "${LMSV_ROOT}/tmp/task6"
TMP_MODEL_CONFIG="${LMSV_ROOT}/tmp/task6/model_config_opensora_$(date +%s).json"

# 调用配置预处理脚本处理模型配置（变异后的配置可能包含占位符）
bash "${SCRIPT_DIR}/prepare_mm_config.sh" "${MM_MODEL}" "${TMP_MODEL_CONFIG}" "opensora"

# 更新为临时文件
export MM_MODEL="${TMP_MODEL_CONFIG}"
echo "Processed model config: ${TMP_MODEL_CONFIG}"
echo "Note: No data_config needed for inference model"

# 设置trap，在脚本退出时删除临时配置文件
cleanup_tmp_config() {
    if [ -f "${TMP_MODEL_CONFIG}" ]; then
        rm -f "${TMP_MODEL_CONFIG}"
        echo "Cleaned up temp config: ${TMP_MODEL_CONFIG}"
    fi
}
trap cleanup_tmp_config EXIT

# 设置MindSpeed-MM路径（必须在conda activate之后设置）
if [ -n "$MINDSPEED_MM_PATH" ]; then
    # 将MINDSPEED_MM_PATH转换为绝对路径（支持相对路径）
    if [[ "$MINDSPEED_MM_PATH" = /* ]]; then
        MM_PATH_ABS="$MINDSPEED_MM_PATH"
    else
        MM_PATH_ABS="$(cd "$MINDSPEED_MM_PATH" && pwd)"
    fi
    # Auto-derive MindSpeed-MM subdirectory if path is workspace root
    if [ ! -f "${MM_PATH_ABS}/pretrain_vlm.py" ] && [ ! -f "${MM_PATH_ABS}/pretrain_sora.py" ]; then
        DERIVED="${MM_PATH_ABS}/MindSpeed-MM"
        if [ -d "$DERIVED" ] && ([ -f "${DERIVED}/pretrain_vlm.py" ] || [ -f "${DERIVED}/pretrain_sora.py" ]); then
            MM_PATH_ABS="$DERIVED"
            echo "Auto-derived MindSpeed-MM path: ${MM_PATH_ABS}"
        fi
    fi
    export MINDSPEED_MM_PATH="$MM_PATH_ABS"
else
    echo "ERROR: MINDSPEED_MM_PATH environment variable not set"
    exit 1
fi
export PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}"
echo "MindSpeed-MM path: ${MINDSPEED_MM_PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"

bash "${SCRIPT_DIR}/pta_opensora_real.sh"
