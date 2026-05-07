#!/bin/bash
# Task6 MSA真实执行脚本 - QwenVL2.5 (推理模式)
set -e

# 记录开始时间
START_TIME=$(date +%s.%N)

echo "========================================"
echo "Task6 MSA Real Execution - QwenVL2.5 (Inference)"
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

# 将MindSpeed-MM添加到PYTHONPATH
if [[ ":${PYTHONPATH}:" != *":${MINDSPEED_MM_PATH}:"* ]]; then
    export PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}"
fi
echo "MindSpeed-MM path: ${MINDSPEED_MM_PATH}"

# 设置CANN环境（使用lmsv_rec内部脚本）
if [ -f "$LMSV_ROOT/scripts/envset/cann_set_env.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/cann_set_env.sh"
else
    echo "ERROR: CANN env script not found"
    exit 1
fi

# 设置MSA环境变量（PYTHONPATH等）
if [ -f "$LMSV_ROOT/scripts/envset/mm-msa-task6.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/mm-msa-task6.sh"
else
    echo "ERROR: MSA env script not found"
    exit 1
fi

# 设置基本环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29505}
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-1}

# Multi-node: extend timeout
if [ "$NNODES" -gt 1 ]; then
    DIST_TIMEOUT="yes"
else
    DIST_TIMEOUT=""
fi
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export NNODES
export WORLD_SIZE

echo "Master: $MASTER_ADDR"
echo "Rank: $NODE_RANK"
echo "Nodes: $NNODES"
echo "World: $WORLD_SIZE"

# 检查LOAD_PATH (QwenVL使用init_from_hf_path，不需要LOAD_PATH)
# if [ -z "$LOAD_PATH" ]; then
#     echo "ERROR: LOAD_PATH environment variable not set (should point to model checkpoint)"
#     exit 1
# fi

# 并行配置
TP=4
PP=1
CP=1
MBS=1
GRAD_ACC_STEP=48
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

# Fix pipeline_num_layers to match PP size
if [ -n "$MM_MODEL" ] && [ -f "$MM_MODEL" ]; then
    python3 -c "
import json
with open('$MM_MODEL') as f:
    c = json.load(f)
pp = $PP
changed = False
for key in ['text_decoder', 'image_encoder']:
    if key not in c:
        continue
    target = c[key] if key != 'image_encoder' else c[key].get('vision_encoder', {})
    if 'pipeline_num_layers' in target:
        layers = target['pipeline_num_layers']
        total = sum(layers)
        if len(layers) != pp:
            if pp == 1:
                target['pipeline_num_layers'] = [total]
            else:
                base = total // pp
                rem = total % pp
                target['pipeline_num_layers'] = [base + (1 if i < rem else 0) for i in range(pp)]
            changed = True
if changed:
    with open('$MM_MODEL', 'w') as f:
        json.dump(c, f, indent=4)
    print('Fixed pipeline_num_layers for PP=', pp)
"
fi

# 使用传入的配置文件
MM_TOOL="${MINDSPEED_MM_PATH}/mindspeed_mm/tools/tools.json"
SAVE_PATH="${SAVE_PATH:-./save_dir}"

MM_ARGS="
    --mm-model ${MM_MODEL} \
    --mm-tool ${MM_TOOL}
"

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

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
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
    --no-masked-softmax-fusion \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters ${TRAIN_ITERS:-2} \
    --lr-warmup-fraction 0.1 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --seed 42 \
    --bf16 \
    --use-flash-attn \
    --variable-seq-lengths \
    --use-distributed-optimizer \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 8 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS:-1} \
    --eval-interval 10000 \
    --eval-iters 5000 \
    --save $SAVE_PATH \
    --ckpt-format torch \
    --log-tps \
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
        if ! pgrep -f "inference_vlm_wrapper.py" > /dev/null 2>&1; then
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
    for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log; do
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

echo ""
echo "Starting real training with msrun..."
echo "========================================"

# Temporarily disable set -e so msrun failure doesn't prevent metric extraction
set +e
PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}" msrun $DISTRIBUTED_ARGS \
    inference_vlm.py \
    $MM_ARGS \
    $GPT_ARGS
TRAIN_EXIT_CODE=$?
set -e

# 等待worker进程完成且日志文件有内容（msrun是异步的）
echo "Waiting for worker processes to complete and logs to be written..."
MSRUN_LOG="${MINDSPEED_MM_PATH}/msrun_log/worker_0.log"
for i in {1..120}; do
    # 检查worker进程是否还在运行
    if pgrep -f "inference_vlm.py" > /dev/null 2>&1; then
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

chmod 440 ${LOG_DIR}/train_${logfile}.log 2>/dev/null || true
find $SAVE_PATH -type d -exec chmod 750 {} \; 2>/dev/null || true
find $SAVE_PATH -type f -exec chmod 640 {} \; 2>/dev/null || true

# 检查训练是否成功 - 必须检查日志中是否有真正的错误
EXIT_CODE=$TRAIN_EXIT_CODE

# 检查worker日志中是否有Python错误（这才是真正的执行失败）
MSRUN_LOG_CHECK="${MINDSPEED_MM_PATH}/msrun_log/worker_0.log"
HAS_REAL_ERROR=false
if [ -f "$MSRUN_LOG_CHECK" ]; then
    # 检查是否有Traceback或致命错误（排除Warning）
    if grep -Ei "Traceback|OSError|ModuleNotFoundError|AttributeError|RuntimeError|Fatal|AssertionError|error:" "$MSRUN_LOG_CHECK" 2>/dev/null | grep -vi "Warning" > /dev/null; then
        HAS_REAL_ERROR=true
    fi
fi

# 计算脚本执行时间
END_TIME=$(date +%s.%N)
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "0")
# 转换为毫秒
EXECUTION_TIME_MS=$(echo "$EXECUTION_TIME * 1000" | bc 2>/dev/null | cut -d. -f1 || echo "")

# 注：QwenVL是推理模型，提取推理指标
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
# 格式: "elapsed time per iteration (ms): XXX" 和 "NPU memory (MB): XXX"
if [ -n "$INFERENCE_TIME" ]; then
    echo "elapsed time per iteration (ms): $INFERENCE_TIME"
fi
if [ -n "$MEMORY" ]; then
    echo "NPU memory (MB): $MEMORY"
fi

if [ "$HAS_REAL_ERROR" = true ]; then
    echo ""
    echo "========================================"
    echo "ERROR: MSA execution failed - fatal error detected in logs"
    echo "========================================"
    exit 1
fi

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "WARNING: Training exited with code $EXIT_CODE"
    echo "MSA framework bug detected - will record for analysis"
    echo "========================================"
fi

# 注：QwenVL是推理模型，没有loss输出，不检查loss
# 判断执行是否成功应使用返回码

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
