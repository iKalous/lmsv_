#!/bin/bash
# Task6 多模态模型MSA环境运行脚本模板
# 用于在MSA环境中执行多模态模型的训练/推理

set -e

# 参数检查
if [ -z "$MM_MODEL" ]; then
    echo "错误: 未设置 MM_MODEL 环境变量"
    exit 1
fi

if [ -z "$MM_DATA" ]; then
    echo "错误: 未设置 MM_DATA 环境变量"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "错误: 未设置 MODEL_NAME 环境变量"
    exit 1
fi

echo "========================================"
echo "Task6 MSA环境运行"
echo "模型: $MODEL_NAME"
echo "MM_MODEL: $MM_MODEL"
echo "MM_DATA: $MM_DATA"
echo "========================================"

# 设置CANN环境
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || true

# 设置Python路径（从环境变量获取）
if [ -z "$MSA_PATH" ]; then
    echo "错误: 未设置 MSA_PATH 环境变量"
    exit 1
fi
MM_PATH="$MSA_PATH"
export PYTHONPATH=${MM_PATH}/MSAdapter:${MM_PATH}/MSAdapter/msa_thirdparty:${MM_PATH}/Megatron-LM:${MM_PATH}/MindSpeed:${MM_PATH}/MindSpeed-MM:$PYTHONPATH

# 进入工作目录
cd "${MM_PATH}/MindSpeed-MM"

# 根据模型选择执行脚本（使用scripts-ms目录下的脚本）
case "$MODEL_NAME" in
    internvl3)
        SCRIPT="scripts-ms/finetune_internvl3_8B.sh"
        ;;
    qwenvl)
        SCRIPT="scripts-ms/finetune_qwen2_5_vl_7b.sh"
        ;;
    opensora)
        SCRIPT="scripts-ms/inference_opensora1_2.sh"
        ;;
    cogvideox)
        SCRIPT="scripts-ms/pretrain_cogvideox_i2v_1.5.sh"
        ;;
    *)
        echo "错误: 不支持的模型: $MODEL_NAME"
        exit 1
        ;;
esac

if [ ! -f "$SCRIPT" ]; then
    echo "错误: 脚本不存在: $SCRIPT"
    exit 1
fi

echo "执行脚本: $SCRIPT"

# 清理之前的msrun_log
rm -rf msrun_log

# 执行训练/推理
bash "$SCRIPT"

echo "========================================"
echo "Task6 MSA环境运行完成"
echo "========================================"
