#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 设置CANN环境（使用lmsv_rec内部脚本）
if [ -f "$LMSV_ROOT/scripts/envset/cann_set_env.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/cann_set_env.sh"
else
    echo "ERROR: CANN env script not found"
    exit 1
fi

# 设置PTA环境变量（包括conda激活）
if [ -f "$LMSV_ROOT/scripts/envset/mm-pta-task6.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/mm-pta-task6.sh"
else
    echo "ERROR: PTA env script not found"
    exit 1
fi
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Align with Task4-5 mm_test.sh for multi-node stability
export NPU_FUSION_OP_ENABLE=0
export NCCL_ALGO=Ring
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$SCRIPT_DIR/cache_ascend}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29505}
NNODES=${NNODES:-1}

# Multi-node: disable overlap to avoid deadlock, extend timeout
if [ "$NNODES" -gt 1 ]; then
    ENABLE_OVERLAP=""
else
    ENABLE_OVERLAP="yes"
fi
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=4
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=4
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

# 设置MindSpeed-MM路径（必须从环境变量获取）
if [ -n "$MINDSPEED_MM_PATH" ]; then
    export MINDSPEED_MM_PATH
else
    echo "ERROR: MINDSPEED_MM_PATH environment variable not set"
    exit 1
fi

# 数据路径从环境变量获取
if [ -z "$MM_DATA" ]; then
    echo "ERROR: MM_DATA environment variable not set"
    exit 1
fi
if [ -z "$MM_MODEL" ]; then
    echo "ERROR: MM_MODEL environment variable not set"
    exit 1
fi
if [ -z "$LOAD_PATH" ]; then
    echo "ERROR: LOAD_PATH environment variable not set (should point to model checkpoint)"
    exit 1
fi

MM_TOOL="${MINDSPEED_MM_PATH}/mindspeed_mm/tools/tools.json"
SAVE_PATH="${SAVE_PATH:-./save_dir}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 1e-4 \
    --lr-warmup-init 1e-5 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters ${TRAIN_ITERS:-10} \
    --no-gradient-accumulation-fusion \
    --load $LOAD_PATH \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 42 \
    --use-distributed-optimizer \
    ${ENABLE_OVERLAP:+--overlap-grad-reduce} \
    ${ENABLE_OVERLAP:+--overlap-param-gather} \
    --distributed-timeout-minutes 30 \
    --allow-tf32 \
    --num-workers 8 \
    --sequence-parallel \
    --qk-layernorm \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS:-1} \
    --eval-interval 10000 \
    --eval-iters 10 \
    --save $SAVE_PATH \
    --ckpt-format torch \
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

# Now run the original pretrain_sora
if __name__ == "__main__":
    # Read and execute pretrain_sora.py
    train_path = os.path.join(ms_path, "pretrain_sora.py")
    with open(train_path, 'r') as f:
        code = compile(f.read(), train_path, 'exec')

    # Create namespace
    namespace = {'__file__': train_path, '__name__': '__main__'}

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

torchrun $DISTRIBUTED_ARGS "$WRAPPER_SCRIPT" \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee ${LOG_DIR}/train_${logfile}.log

EXIT_CODE=${PIPESTATUS[0]}
rm -f "$WRAPPER_SCRIPT"

# 检查训练是否成功
if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: PTA training failed with exit code $EXIT_CODE"
    echo "========================================"
    exit 1
fi

# 检查日志中是否有真正的错误标记（排除Warning）
if grep -Ei "Fatal|RuntimeError|AssertionError|Traceback" ${LOG_DIR}/train_${logfile}.log | grep -vi "Warning" > /dev/null; then
    echo ""
    echo "========================================"
    echo "ERROR: Training failed - fatal error detected in logs"
    echo "========================================"
    exit 1
fi

chmod 440 ${LOG_DIR}/train_${logfile}.log
find $SAVE_PATH -type d -exec chmod 750 {} \;
find $SAVE_PATH -type f -exec chmod 640 {} \;

# 提取关键指标 - 必须真实测到，不允许默认值
STEP_TIME=`grep "elapsed time per iteration" ${LOG_DIR}/train_*.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`

# 验证提取的指标
if [ -z "$STEP_TIME" ]; then
    # 多机模式下，迭代日志可能只出现在负责日志输出的节点上，当前节点可能无法提取到
    if [ "$NNODES" -gt 1 ]; then
        echo ""
        echo "========================================"
        echo "WARNING: Failed to extract step time from training logs (normal in multi-node)"
        echo "========================================"
        STEP_TIME="N/A"
    else
        echo ""
        echo "========================================"
        echo "ERROR: Failed to extract step time from training logs"
        echo "========================================"
        exit 1
    fi
fi

if [ "$STEP_TIME" = "N/A" ]; then
    SPS="N/A"
else
    SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
fi
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $SPS"

echo ""
echo "========================================"
echo "PTA training completed successfully"
echo "========================================"
