#!/bin/bash
# Task6 PTA真实执行脚本 - QwenVL2.5 (推理模式)
set -e

# 记录开始时间
START_TIME=$(date +%s.%N)

echo "========================================"
echo "Task6 PTA Real Execution - QwenVL2.5 (Inference)"
echo "MM_MODEL: $MM_MODEL"
echo "========================================"

# 检查必需环境变量
if [ -z "$MM_MODEL" ]; then
    echo "ERROR: MM_MODEL environment variable not set"
    exit 1
fi

if [ ! -f "$MM_MODEL" ]; then
    echo "ERROR: Model config not found: $MM_MODEL"
    exit 1
fi

echo "Configuration check PASSED"
echo "Using mutated config: $MM_MODEL"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 设置MindSpeed-MM路径（必须从环境变量获取）
if [ -n "$MINDSPEED_MM_PATH" ]; then
    export MINDSPEED_MM_PATH
else
    echo "ERROR: MINDSPEED_MM_PATH environment variable not set"
    exit 1
fi

# 将MindSpeed-MM添加到PYTHONPATH（如果入口脚本未设置）
if [[ ":${PYTHONPATH}:" != *":${MINDSPEED_MM_PATH}:"* ]]; then
    export PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}"
fi
echo "MindSpeed-MM path: ${MINDSPEED_MM_PATH}"

# 设置CANN环境（使用lmsv_rec内部脚本）
if [ -f "$LMSV_ROOT/scripts/envset/cann_set_env.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/cann_set_env.sh"
else
    echo "ERROR: CANN env script not found at $LMSV_ROOT/scripts/envset/cann_set_env.sh"
    exit 1
fi

# 设置PTA环境变量（PYTHONPATH等）
if [ -f "$LMSV_ROOT/scripts/envset/mm-pta-task6.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/mm-pta-task6.sh"
else
    echo "ERROR: PTA env script not found"
    exit 1
fi

source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Align with Task4-5 mm_test.sh for multi-node stability
export NPU_FUSION_OP_ENABLE=0
export NCCL_ALGO=Ring
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$SCRIPT_DIR/cache_ascend}"

# 推理使用单卡
NPUS_PER_NODE=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=6000
NNODES=${NNODES:-1}

# Multi-node: extend timeout
if [ "$NNODES" -gt 1 ]; then
    DIST_TIMEOUT="yes"
else
    DIST_TIMEOUT=""
fi
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 使用传入的配置文件（来自lmsv_rec内部）
TP=1
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 检查是否使用init_from_hf_path模式（HF格式权重）
# 如果MM_MODEL中包含init_from_hf_path，则不使用--load参数
if [ -n "$LOAD_PATH" ] && [ -f "$LOAD_PATH/latest_checkpointed_iteration.txt" ]; then
    echo "Using MindSpeed-MM format checkpoint: $LOAD_PATH"
    LOAD_ARG="--load $LOAD_PATH"
else
    echo "Using HuggingFace format checkpoint via init_from_hf_path"
    LOAD_ARG=""
fi

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size 152064 \
    --seq-length 1024 \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --seed 42 \
    --bf16 \
    --use-flash-attn \
    --no-load-optim \
    --no-load-rng
    ${LOAD_ARG}
"

MM_ARGS="
    --mm-model $MM_MODEL
"

SAVE_PATH="${SAVE_PATH:-./save_dir}"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS:-1} \
    --eval-interval 10000 \
    --eval-iters 5000 \
    --save $SAVE_PATH \
    --ckpt-format torch
"

# PTA日志目录
LOG_DIR="${LMSV_ROOT}/pta_logs"
mkdir -p ${LOG_DIR}

logfile=$(date +%Y%m%d)_$(date +%H%M%S)_rank${NODE_RANK:-0}

# Create a wrapper script that patches memory reporting
WRAPPER_SCRIPT="${LOG_DIR}/pta_memory_wrapper_${logfile}.py"
cat > "$WRAPPER_SCRIPT" << 'EOF'
import sys
import os

# Add MindSpeed-MM to path first
ms_path = os.environ.get('MINDSPEED_MM_PATH')
if not ms_path:
    raise RuntimeError("MINDSPEED_MM_PATH environment variable is not set")
if os.path.exists(ms_path):
    sys.path.insert(0, ms_path)

# Import torch and torch_npu
import torch
import torch_npu

# Reset memory stats before run
try:
    torch_npu.npu.reset_peak_memory_stats()
except:
    pass

# Now run the original inference_vlm
if __name__ == "__main__":
    # Read and execute inference_vlm.py
    inference_path = os.path.join(ms_path, "inference_vlm.py")
    with open(inference_path, 'r') as f:
        code = compile(f.read(), inference_path, 'exec')

    # Create namespace
    namespace = {'__file__': inference_path, '__name__': '__main__'}

    try:
        exec(code, namespace)
    finally:
        # Print memory info regardless of success/failure
        try:
            max_mem = torch_npu.npu.max_memory_allocated() / (1024 ** 2)
            print(f"\n========================================", flush=True)
            print(f"NPU memory (MB): {max_mem:.2f}", flush=True)
            print(f"max allocated: {max_mem:.2f}", flush=True)
            print(f"========================================", flush=True)
        except Exception as e:
            print(f"\n[ERROR] Failed to get memory: {e}", flush=True)
EOF

chmod +x "$WRAPPER_SCRIPT"

# 进入MindSpeed-MM目录执行
cd ${MINDSPEED_MM_PATH}

# Run torchrun with the wrapper script
PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}" torchrun $DISTRIBUTED_ARGS \
    "$WRAPPER_SCRIPT" \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee ${LOG_DIR}/inference_${logfile}.log

INFERENCE_EXIT_CODE=${PIPESTATUS[0]}
rm -f "$WRAPPER_SCRIPT"


chmod 440 ${LOG_DIR}/inference_${logfile}.log 2>/dev/null || true

# 检查推理是否成功
if [ "$INFERENCE_EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: PTA inference failed with exit code $INFERENCE_EXIT_CODE"
    echo "========================================"
    exit 1
fi

# 检查日志中是否有真正的错误标记（排除Warning）
if grep -Ei "Fatal|RuntimeError|AssertionError" ${LOG_DIR}/inference_${logfile}.log 2>/dev/null | grep -vi "Warning" > /dev/null; then
    echo ""
    echo "========================================"
    echo "ERROR: Inference failed - fatal error detected in logs"
    echo "========================================"
    exit 1
fi

# 计算脚本执行时间
END_TIME=$(date +%s.%N)
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "0")
# 转换为毫秒
EXECUTION_TIME_MS=$(echo "$EXECUTION_TIME * 1000" | bc 2>/dev/null | cut -d. -f1 || echo "")

# 提取关键指标 - 推理模式
# 尝试从MindSpeed-MM日志中提取elapsed time per iteration
INFERENCE_TIME=$(grep -E "elapsed time per iteration" ${LOG_DIR}/inference_${logfile}.log 2>/dev/null | tail -1 | grep -oP '[\d.]+' 2>/dev/null | head -1 || echo "")
# 如果没有，使用脚本执行时间
if [ -z "$INFERENCE_TIME" ] && [ -n "$EXECUTION_TIME_MS" ]; then
    INFERENCE_TIME="$EXECUTION_TIME_MS"
fi

# 提取显存 - 尝试从日志中提取，如果没有则通过torch_npu获取
MEMORY=$(grep -E "(NPU memory|memory \(MB\))" ${LOG_DIR}/inference_${logfile}.log 2>/dev/null | tail -1 | grep -oP 'max allocated:\s*\K[\d.]+' 2>/dev/null || echo "")

# 如果日志中没有显存信息，通过Python获取
if [ -z "$MEMORY" ]; then
    MEMORY=$(python3 -c "import torch_npu; print(f'{torch_npu.npu.max_memory_allocated() / 1024**2:.2f}')" 2>/dev/null || echo "")
fi

# 输出与训练模型相同格式的指标，便于task6解析
# 格式: "elapsed time per iteration (ms): XXX" 和 "NPU memory (MB): XXX"
echo ""
echo "========================================"
echo "PTA inference completed successfully"
echo "Inference execution metrics:"
if [ -n "$INFERENCE_TIME" ]; then
    echo "elapsed time per iteration (ms): $INFERENCE_TIME"
fi
if [ -n "$MEMORY" ]; then
    echo "NPU memory (MB): $MEMORY"
fi
echo "========================================"
