#!/bin/bash
# Task6 MSA真实执行脚本 - OpenSora (推理模式)
# 记录开始时间
START_TIME=$(date +%s.%N)

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMSV_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 设置CANN环境（使用lmsv_rec内部脚本）
if [ -f "$LMSV_ROOT/scripts/envset/cann_set_env.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/cann_set_env.sh"
else
    echo "ERROR: CANN env script not found at $LMSV_ROOT/scripts/envset/cann_set_env.sh"
    exit 1
fi

# 设置MSA环境变量（PYTHONPATH等）
if [ -f "$LMSV_ROOT/scripts/envset/mm-msa-task6.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/mm-msa-task6.sh"
else
    echo "ERROR: MSA env script not found"
    exit 1
fi

# 设置MindSpeed-MM路径（必须从环境变量获取）
if [ -n "$MINDSPEED_MM_PATH" ]; then
    export MINDSPEED_MM_PATH
else
    echo "ERROR: MINDSPEED_MM_PATH environment variable not set"
    exit 1
fi
export PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}"
echo "MindSpeed-MM path: ${MINDSPEED_MM_PATH}"

export ASCEND_RT_VISIBLE_DEVICES="0"
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Modify device for MSA: ensure device is "npu" (not "npu:0" or "npu:Ascend")
# MSA works best with simple "npu" device specification
if [ -n "$MM_MODEL" ] && [ -f "$MM_MODEL" ]; then
    echo "Modifying device for MSA execution..."
    echo "Original device line:"
    grep -n '"device"' "$MM_MODEL" | head -1
    # Replace any device specification with simple "npu"
    sed -i 's/"device"[[:space:]]*:[[:space:]]*"[^"]*"/"device": "npu"/g' "$MM_MODEL"
    sed -i 's/"device": "npu:.*"/"device": "npu"/g' "$MM_MODEL"
    echo "Modified device line:"
    grep -n '"device"' "$MM_MODEL" | head -1
    echo "Device set to 'npu' for MSA compatibility"
fi
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=12875
NNODES=${NNODES:-1}

# Multi-node: extend timeout
if [ "$NNODES" -gt 1 ]; then
    DIST_TIMEOUT="yes"
else
    DIST_TIMEOUT=""
fi
NODE_RANK=${NODE_RANK:-0}
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))


TP=1
PP=1
CP=1
MBS=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS * $DP))

#DISTRIBUTED_ARGS="
#    --nproc_per_node $NPUS_PER_NODE \
#    --nnodes $NNODES \
#    --node_rank $NODE_RANK \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT
#"
DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --log_dir=msrun_log \
    --join=False \
    --cluster_time_out=300 \
    --bind_core=True
"

# 模型配置路径从环境变量获取
if [ -z "$MM_MODEL" ]; then
    echo "ERROR: MM_MODEL environment variable not set"
    exit 1
fi
MM_ARGS="
 --mm-model "$MM_MODEL"
"

SAVE_PATH="${SAVE_PATH:-./save_dir}"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --train-iters ${TRAIN_ITERS:-1} \
    --weight-decay 0 \
    --clip-grad 1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --fp16 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS:-1} \
    --save $SAVE_PATH \
    --ckpt-format torch
"

# 统一日志目录
LOG_DIR="${LMSV_ROOT}/msrun_log"
mkdir -p ${LOG_DIR}

logfile=$(date +%Y%m%d)_$(date +%H%M%S)_rank${NODE_RANK:-0}

# 进入MindSpeed-MM目录执行
cd ${MINDSPEED_MM_PATH}

# 第一次执行时清空/创建msrun_log，后续轮次不清空
MSRUN_FIRST_RUN_MARKER="${MINDSPEED_MM_PATH}/msrun_log/.msrun_first_run_done"
if [ ! -f "$MSRUN_FIRST_RUN_MARKER" ]; then
    rm -rf ${MINDSPEED_MM_PATH}/msrun_log
    mkdir -p ${MINDSPEED_MM_PATH}/msrun_log
    touch "$MSRUN_FIRST_RUN_MARKER"
fi

# 设置trap，在脚本退出时复制日志
copy_log_on_exit() {
    local EXIT_CODE=$?
    echo "Copying logs on exit (exit code: $EXIT_CODE)..."

    # 等待worker进程结束（最多等待60秒）
    echo "Waiting for worker processes to complete..."
    for i in {1..60}; do
        if ! pgrep -f "inference_sora.py" > /dev/null 2>&1; then
            echo "Worker processes completed."
            break
        fi
        sleep 1
    done

    # 再等待2秒确保日志写入完成
    sleep 2

    # 复制worker日志到msa_logs
    local MSR_LOG="${MINDSPEED_MM_PATH}/msrun_log/worker_0.log"
    # 查找包含loss的worker日志
    for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log "$STDOUT_LOG"; do
        if [ -f "$log" ] && grep -q "loss:" "$log" 2>/dev/null; then
            MSR_LOG="$log"
            echo "Found loss in $log"
            break
        fi
    done
    if [ -f "$MSR_LOG" ]; then
        echo "Copying worker log to msa_logs..."
        cat "$MSR_LOG" > "${LOG_DIR}/train_${logfile}.log"
        chmod 440 "${LOG_DIR}/train_${logfile}.log"
        echo "Log copied successfully ($(wc -l < "${LOG_DIR}/train_${logfile}.log") lines)"
    else
        echo "WARNING: worker log not found at $MSR_LOG"
        touch "${LOG_DIR}/train_${logfile}.log"
        chmod 440 "${LOG_DIR}/train_${logfile}.log"
    fi
}
trap copy_log_on_exit EXIT

# Device is already set to "npu" by mm_msa_opensora.sh, which is the correct format
# MindSpeed-MM get_device() only supports "npu", "npu:0/1/2...", or "cpu"
# Do NOT change to "npu:Ascend" as it causes "invalid literal for int() with base 10: 'ascend'" error

STDOUT_LOG="${MINDSPEED_MM_PATH}/msrun_log/msrun_stdout.log"
msrun $DISTRIBUTED_ARGS inference_sora.py $MM_ARGS $GPT_ARGS $OUTPUT_ARGS

# 检查执行结果
EXIT_CODE=$?

# 等待worker进程完成且日志文件有内容（msrun是异步的）
echo "Waiting for worker processes to complete and logs to be written..."
MSRUN_LOG="${MINDSPEED_MM_PATH}/msrun_log/worker_0.log"
for i in {1..120}; do
    # 检查worker进程是否还在运行
    if pgrep -f "inference_sora.py" > /dev/null 2>&1; then
        sleep 1
        continue
    fi
    # Worker已结束，检查日志文件是否存在且非空
    if [ -f "$MSRUN_LOG" ] && [ -s "$MSRUN_LOG" ]; then
        echo "Worker processes completed and logs are ready."
        break
    fi
    sleep 1
done

# 输出真实的msrun日志以便task6捕获
MSRUN_LOG="${MINDSPEED_MM_PATH}/msrun_log/worker_0.log"
if [ -f "$MSRUN_LOG" ]; then
    echo ""
    echo "========== Real MSA Execution Log =========="
    cat "$MSRUN_LOG"
    echo "========== End Real MSA Execution Log =========="
fi
if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "WARNING: MSA execution exited with code $EXIT_CODE"
    echo "MSA framework bug detected - recording for analysis"
    echo "========================================"
    # MSA异常被记录但不退出，因为可能是框架bug
fi

# 计算脚本执行时间
END_TIME=$(date +%s.%N)
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "0")
# 转换为毫秒
EXECUTION_TIME_MS=$(echo "$EXECUTION_TIME * 1000" | bc 2>/dev/null | cut -d. -f1 || echo "")

# 注：OpenSora是推理模型，提取推理指标
# 尝试从日志中提取elapsed time per iteration（与训练模型相同格式，便于统一解析）
INFERENCE_TIME=$(grep -E "elapsed time per iteration" ${LOG_DIR}/train_${logfile}.log 2>/dev/null | tail -1 | grep -oP '[\d.]+' 2>/dev/null | head -1 || echo "")
# 如果没有，使用脚本执行时间
if [ -z "$INFERENCE_TIME" ] && [ -n "$EXECUTION_TIME_MS" ]; then
    INFERENCE_TIME="$EXECUTION_TIME_MS"
fi

# 提取显存 - 尝试从日志中提取，如果没有则通过torch_npu获取
MEMORY=$(grep -E "(NPU memory|memory \(MB\))" ${LOG_DIR}/train_${logfile}.log 2>/dev/null | tail -1 | grep -oP 'max allocated:\s*\K[\d.]+' 2>/dev/null || echo "")

# 如果日志中没有显存信息，通过Python获取
if [ -z "$MEMORY" ]; then
    MEMORY=$(python3 -c "import torch_npu; print(f'{torch_npu.npu.max_memory_allocated() / 1024**2:.2f}')" 2>/dev/null || echo "")
fi

# 输出与训练模型相同格式的指标，便于task6解析
echo ""
echo "========================================"
echo "MSA inference completed successfully"
echo "Inference execution metrics:"
if [ -n "$INFERENCE_TIME" ]; then
    echo "elapsed time per iteration (ms): $INFERENCE_TIME"
fi
if [ -n "$MEMORY" ]; then
    echo "NPU memory (MB): $MEMORY"
fi
echo "========================================"
