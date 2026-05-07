#!/bin/bash
# 多模态配置文件预处理脚本
# 将配置文件中的占位符替换为实际路径

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_config> <output_config> [model_type]"
    echo "  model_type: internvl3, qwenvl, opensora, cogvideox"
    exit 1
fi

INPUT_CONFIG="$1"
OUTPUT_CONFIG="$2"
MODEL_TYPE="${3:-internvl3}"
ENV_TYPE="${4:-pta}"

# 检查输入文件
if [ ! -f "$INPUT_CONFIG" ]; then
    echo "ERROR: Input config not found: $INPUT_CONFIG"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_CONFIG")"

# 检查必需的环境变量
if [ -z "$DATASET_PATH" ] && [ -z "$LOAD_PATH" ]; then
    echo "WARNING: Neither DATASET_PATH nor LOAD_PATH is set"
    echo "Configuration may contain unresolved placeholders"
fi

# 检查数据集根目录环境变量（必须设置，不可硬编码）
if [ -z "$DATASET_ROOT" ]; then
    echo "ERROR: DATASET_ROOT environment variable is not set"
    echo "Please set it in config.json or export DATASET_ROOT=/path/to/dataset"
    exit 1
fi

# 根据模型类型设置默认路径
if [ "$MODEL_TYPE" == "internvl3" ]; then
    DATASET_PATH_DEFAULT="${DATASET_PATH:-${DATASET_ROOT}/internvl3}"
    LOAD_PATH_DEFAULT="${LOAD_PATH:-${DATASET_ROOT}/internvl3/raw_ckpt/InternVL3-8B}"
elif [ "$MODEL_TYPE" == "qwenvl" ]; then
    DATASET_PATH_DEFAULT="${DATASET_PATH:-${DATASET_ROOT}/qwen2.5vl}"
    LOAD_PATH_DEFAULT="${LOAD_PATH:-${DATASET_ROOT}/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct}"
elif [ "$MODEL_TYPE" == "opensora" ]; then
    DATASET_PATH_DEFAULT="${DATASET_PATH:-${DATASET_ROOT}/opensora1.2}"
    LOAD_PATH_DEFAULT="${LOAD_PATH:-${DATASET_ROOT}/opensora1.2}"
elif [ "$MODEL_TYPE" == "cogvideox" ]; then
    DATASET_PATH_DEFAULT="${DATASET_PATH:-${DATASET_ROOT}/cogvideox}"
    LOAD_PATH_DEFAULT="${LOAD_PATH:-${DATASET_ROOT}/cogvideox/CogVideoX-5B}"
else
    DATASET_PATH_DEFAULT="${DATASET_PATH:-${DATASET_ROOT}}"
    LOAD_PATH_DEFAULT="${LOAD_PATH:-${DATASET_ROOT}}"
fi

# 使用环境变量或默认值
DATASET_PATH="${DATASET_PATH:-$DATASET_PATH_DEFAULT}"
LOAD_PATH="${LOAD_PATH:-$LOAD_PATH_DEFAULT}"

# 推理步数（用于OpenSora等推理模型）
INFERENCE_STEPS="${TRAIN_ITERS:-5}"
echo "  INFERENCE_STEPS: $INFERENCE_STEPS"

echo "Preparing config for model: $MODEL_TYPE"
echo "  DATASET_ROOT: $DATASET_ROOT"
echo "  DATASET_PATH: $DATASET_PATH"
echo "  LOAD_PATH: $LOAD_PATH"
echo "  Input: $INPUT_CONFIG"
echo "  Output: $OUTPUT_CONFIG"

# 对于OpenSora，使用单prompt文件
if [ "$MODEL_TYPE" == "opensora" ]; then
    SINGLE_PROMPT="/tmp/opensora_single_prompt.txt"
    # 只保留第一个prompt（第一行）
    head -1 "${DATASET_PATH}/samples_prompts.txt" > "$SINGLE_PROMPT" 2>/dev/null || echo "A serene scene." > "$SINGLE_PROMPT"
fi

# 替换占位符
sed -e "s|{{DATASET_PATH}}|${DATASET_PATH}|g" \
    -e "s|{{LOAD_PATH}}|${LOAD_PATH}|g" \
    -e "s|{{DATASET_ROOT}}|${DATASET_ROOT}|g" \
    -e "s|examples/opensora1.2/samples_prompts.txt|/tmp/opensora_single_prompt.txt|g" \
    "$INPUT_CONFIG" > "$OUTPUT_CONFIG"

# 单独处理 num_inference_steps，避免重复替换
if grep -q '"num_inference_steps"' "$OUTPUT_CONFIG" 2>/dev/null; then
    # 使用临时文件，精确替换数字部分
    sed -i -E 's/"num_inference_steps":[[:space:]]*[0-9]+/"num_inference_steps":'"${INFERENCE_STEPS}"'/g' "$OUTPUT_CONFIG"
fi

# 对于CogVideoX，修复decord兼容性问题（NPU环境下decord.cpu属性不存在）
if [ "$MODEL_TYPE" == "cogvideox" ]; then
    if grep -q 'DecordVideo' "$OUTPUT_CONFIG" 2>/dev/null; then
        # 将DecordVideo替换为TorchvisionVideo以避免decord库问题
        # TorchvisionVideo在registry中可用，而OpenCVVideo不可用
        sed -i 's/DecordVideo/TorchvisionVideo/g' "$OUTPUT_CONFIG"
        echo "  Fixed: Changed video_reader_type from DecordVideo to TorchvisionVideo for NPU compatibility"
    fi
fi

# 注意：dtype格式处理
# PTA和MSA都使用MindSpeed-MM加载配置，统一支持bf16/fp16/fp32简写格式
# transformers的dict_dtype_to_str函数已通过patch支持短格式

echo "Config prepared successfully: $OUTPUT_CONFIG"
