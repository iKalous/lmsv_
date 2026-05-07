#!/bin/bash
# Task6 MSA真实执行脚本 - InternVL3
# 必须真实执行模型训练
set -e


echo "========================================"
echo "Task6 MSA Real Execution - InternVL3"
echo "MM_MODEL: $MM_MODEL"
echo "MM_DATA: $MM_DATA"
echo "========================================"

# 检查必需环境变量
if [ -z "$MM_MODEL" ]; then
    echo "ERROR: MM_MODEL environment variable not set"
    exit 1
fi
if [ -z "$MM_DATA" ]; then
    echo "ERROR: MM_DATA environment variable not set"
    exit 1
fi

if [ ! -f "$MM_MODEL" ]; then
    echo "ERROR: Model config not found: $MM_MODEL"
    exit 1
fi

echo "Configuration check PASSED"
echo "Using mutated config: $MM_MODEL"
echo "Using data config: $MM_DATA"

# 设置CANN环境
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

# 设置MSA环境变量（PYTHONPATH等）
if [ -f "$LMSV_ROOT/scripts/envset/mm-msa-task6.sh" ]; then
    source "$LMSV_ROOT/scripts/envset/mm-msa-task6.sh"
else
    echo "ERROR: MSA env script not found"
    exit 1
fi

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000

# MSA使用单机配置
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6002}
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

# 并行配置
MBS=1
GRAD_ACC_STEP=64
TP=1
PP=4
CP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

# 设置MindSpeed-MM路径（必须从环境变量获取）
if [ -n "$MINDSPEED_MM_PATH" ]; then
    export MINDSPEED_MM_PATH
else
    echo "ERROR: MINDSPEED_MM_PATH environment variable not set"
    exit 1
fi

# 使用传入的配置文件
MM_TOOL="${MINDSPEED_MM_PATH}/mindspeed_mm/tools/tools.json"
# 数据集和权重路径从环境变量获取
if [ -z "$LOAD_PATH" ]; then
    echo "ERROR: LOAD_PATH environment variable not set (should point to model checkpoint)"
    exit 1
fi
SAVE_PATH="${SAVE_PATH:-./save_dir}"

MM_ARGS="
    --mm-data ${MM_DATA} \
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
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 4096 \
    --tokenizer-type NullTokenizer \
    --vocab-size 151674 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 2e-5 \
    --min-lr 0.0 \
    --train-iters ${TRAIN_ITERS:-2} \
    --lr-decay-style cosine \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \

    --use-flash-attn \
    --bf16 \
    --load $LOAD_PATH \
    --variable-seq-lengths \
    --normalization RMSNorm \
    --num-workers 4 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS:-1} \
    --eval-interval 5000 \
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

echo ""
echo "Starting real training with msrun..."
echo "========================================"

# 第一次执行时清空/创建msrun_log，后续轮次不清空
MSRUN_FIRST_RUN_MARKER="${MINDSPEED_MM_PATH}/msrun_log/.msrun_first_run_done"
if [ ! -f "$MSRUN_FIRST_RUN_MARKER" ]; then
    if [ "${NODE_RANK:-0}" -eq 0 ]; then
        rm -rf ${MINDSPEED_MM_PATH}/msrun_log
    fi
    mkdir -p ${MINDSPEED_MM_PATH}/msrun_log
    touch "$MSRUN_FIRST_RUN_MARKER"
fi

# 设置trap，在脚本退出时复制日志
copy_log_on_exit() {
    local EXIT_CODE=$?
    # 临时禁用errexit，避免pgrep等命令返回非零导致退出
    set +e
    echo "Copying logs on exit (exit code: $EXIT_CODE)..."

    # 等待worker进程结束（最多等待60秒）
    echo "Waiting for worker processes to complete..."
    for i in {1..60}; do
        if ! pgrep -f "pretrain_vlm.py" > /dev/null 2>&1; then
            echo "Worker processes completed."
            break
        fi
        sleep 1
    done

    # 再等待2秒确保日志写入完成
    sleep 2

    # 复制worker日志到msa_logs - 查找包含loss的worker日志
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

STDOUT_LOG="${MINDSPEED_MM_PATH}/msrun_log/msrun_stdout.log"
msrun $DISTRIBUTED_ARGS \
    pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl > "$STDOUT_LOG" 2>&1

TRAIN_EXIT_CODE=$?

# 临时禁用errexit，避免后续grep等命令返回非零导致退出
set +e

# 等待worker进程完成且日志文件有内容（msrun是异步的）
echo "Waiting for worker processes to complete and logs to be written..."
# 查找包含loss的worker日志（PP配置下loss只在最后一个worker输出）
MSRUN_LOG="${MINDSPEED_MM_PATH}/msrun_log/worker_0.log"
LOSS_FOUND=0
PROCESS_ENDED=0
CHECK_COUNT=0
MAX_CHECKS=360

while [ $CHECK_COUNT -lt $MAX_CHECKS ]; do
    CHECK_COUNT=$((CHECK_COUNT + 1))

    # 检查worker进程是否还在运行
    if [ "$PROCESS_ENDED" -eq 0 ] && ! pgrep -f "pretrain_vlm.py" > /dev/null 2>&1; then
        PROCESS_ENDED=1
        echo "Worker processes ended, searching for loss in logs..."
    fi

    # 每次循环都查找包含loss的日志文件
    for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log "$STDOUT_LOG"; do
        if [ -f "$log" ] && [ -s "$log" ]; then
            if grep -q "loss:" "$log" 2>/dev/null; then
                MSRUN_LOG="$log"
                echo "Found loss in $log"
                LOSS_FOUND=1
            fi
            if [ "$MSRUN_LOG" = "${MINDSPEED_MM_PATH}/msrun_log/worker_0.log" ]; then
                MSRUN_LOG="$log"
            fi
        fi
    done

    # 进程结束且找到loss，退出等待
    if [ "$PROCESS_ENDED" -eq 1 ] && [ "$LOSS_FOUND" -eq 1 ]; then
        echo "Loss found and processes ended, exiting wait loop."
        break
    fi
    
    # 如果进程已结束且一直没找到loss，且达到最大检查次数，退出
    if [ "$PROCESS_ENDED" -eq 1 ] && [ "$LOSS_FOUND" -eq 0 ] && [ "$CHECK_COUNT" -ge "$MAX_CHECKS" ]; then
        echo "Timeout waiting for loss, using available log."
        break
    fi

    sleep 1
done


# 额外等待：确保远程worker日志完全刷新，且找到包含所有iteration的日志
echo "Stabilizing: waiting for all iterations to be written to logs..."
FINAL_WAIT_COUNT=0
MAX_FINAL_WAIT=60
TARGET_ITERATIONS=${TRAIN_ITERS:-10}
while [ "$FINAL_WAIT_COUNT" -lt "$MAX_FINAL_WAIT" ]; do
    FINAL_WAIT_COUNT=$((FINAL_WAIT_COUNT + 1))
    
    CURRENT_BEST_COUNT=0
    for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log "$STDOUT_LOG"; do
        if [ -f "$log" ]; then
            COUNT=$(grep -c "loss:" "$log" 2>/dev/null)
            if [ "$COUNT" -gt "$CURRENT_BEST_COUNT" ]; then
                CURRENT_BEST_COUNT=$COUNT
            fi
        fi
    done
    
    if [ "$CURRENT_BEST_COUNT" -ge "$TARGET_ITERATIONS" ]; then
        echo "Found log with $CURRENT_BEST_COUNT/$TARGET_ITERATIONS loss entries"
        break
    fi
    
    if [ "$FINAL_WAIT_COUNT" -ge "$MAX_FINAL_WAIT" ]; then
        echo "Warning: timeout waiting for all iterations, best count: $CURRENT_BEST_COUNT"
        break
    fi
    
    sleep 1
done
# 选择包含最多loss行的worker日志（确保获取最后一个iteration的loss）
BEST_LOG=""
BEST_LOSS_COUNT=0
for log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log "$STDOUT_LOG"; do
    if [ -f "$log" ]; then
        LOSS_COUNT=$(grep -c "loss:" "$log" 2>/dev/null)
        if [ "$LOSS_COUNT" -gt "$BEST_LOSS_COUNT" ]; then
            BEST_LOSS_COUNT=$LOSS_COUNT
            BEST_LOG="$log"
        fi
    fi
done
if [ -n "$BEST_LOG" ]; then
    MSRUN_LOG="$BEST_LOG"
    echo "Selected worker log with most loss entries: $MSRUN_LOG ($BEST_LOSS_COUNT loss lines)"
fi
# 输出真实的msrun日志以便task6捕获
if [ -f "$MSRUN_LOG" ]; then
    echo ""
    echo "========== Real MSA Execution Log =========="
    cat "$MSRUN_LOG"
    echo "========== End Real MSA Execution Log =========="
fi

chmod 440 ${LOG_DIR}/train_${logfile}.log 2>/dev/null || true
find $SAVE_PATH -type d -exec chmod 750 {} \; 2>/dev/null || true
find $SAVE_PATH -type f -exec chmod 640 {} \; 2>/dev/null || true

# 检查训练是否成功
EXIT_CODE=$TRAIN_EXIT_CODE
if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "WARNING: Training exited with code $EXIT_CODE"
    echo "MSA framework bug detected - will record for analysis"
    echo "========================================"
fi

# 提取关键指标 - 直接从MSRUN_LOG源文件提取（copy_log_on_exit还没执行）
STEP_TIME=$(grep "elapsed time per iteration" "$MSRUN_LOG" 2>/dev/null | awk -F '[:,|]' '{print$5}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR>0) print sum/NR; else print "N/A"}')

# 提取loss - 从包含loss的worker日志提取
LOSS=$(grep "loss:" "$MSRUN_LOG" 2>/dev/null | tail -1 | grep -oP 'loss:\s+\K[\d.E+-]+' 2>/dev/null || echo "")

# 提取显存 - 从所有worker日志中提取（memory可能在不同worker中）
# 优先从worker_0获取，如果不存在则从任何包含memory的worker获取
MEMORY=""
for mem_log in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log; do
    if [ -f "$mem_log" ]; then
        MEM_VAL=$(grep -E "memory \(MB\)" "$mem_log" 2>/dev/null | grep -oP 'max allocated:\s*\K[\d.]+' | tail -1)
        if [ -n "$MEM_VAL" ]; then
            MEMORY="$MEM_VAL"
            # 如果是worker_0，优先使用并停止查找
            if [[ "$mem_log" == *"worker_0.log"* ]]; then
                break
            fi
        fi
    fi
done

# 检查是否有真实错误（优先显示真实错误，而非"No loss found"）
HAS_REAL_ERROR=false
for wlog in ${MINDSPEED_MM_PATH}/msrun_log/worker_*.log; do
    if [ -f "$wlog" ] && grep -Eqi "Traceback|RuntimeError|ValueError|TypeError|OSError|ModuleNotFoundError|AttributeError|AssertionError|Fatal" "$wlog" 2>/dev/null; then
        HAS_REAL_ERROR=true
        REAL_ERROR=$(grep -Ei "(RuntimeError|ValueError|TypeError|OSError|ModuleNotFoundError|AttributeError|AssertionError|Fatal).*" "$wlog" 2>/dev/null | grep -vi "Warning" | head -1)
        break
    fi
done

# 检查是否有loss输出（MSA可能没有完整loss记录）
if [ -z "$LOSS" ]; then
    if [ "$HAS_REAL_ERROR" = true ]; then
        echo ""
        echo "========================================"
        echo "ERROR: MSA execution failed"
        echo "$REAL_ERROR"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "WARNING: No loss found in logs - MSA may have crashed early"
        echo "========================================"
    fi
fi

# 记录实际结果
echo ""
echo "========================================"
if [ -n "$LOSS" ] && [ -n "$MEMORY" ]; then
    echo "Training metrics extracted successfully"
    echo "loss: $LOSS"
    echo "NPU memory: $MEMORY MB"
    echo "elapsed time per iteration: $STEP_TIME ms"
else
    echo "Training metrics incomplete"
    echo "loss: ${LOSS:-N/A}"
    echo "NPU memory: ${MEMORY:-N/A} MB"
    echo "elapsed time per iteration: $STEP_TIME ms"
fi
echo "========================================"
