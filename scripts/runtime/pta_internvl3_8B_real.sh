#!/bin/bash
# Task6 PTA真实执行脚本 - InternVL3
# 必须真实执行模型训练
set -e

echo "========================================"
echo "Task6 PTA Real Execution - InternVL3"
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

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Align with Task4-5 mm_test.sh for multi-node stability
export NPU_FUSION_OP_ENABLE=0
export NCCL_ALGO=Ring
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$SCRIPT_DIR/cache_ascend}"
export ACLNN_CACHE_LIMIT=100000

# 固定单节点8卡配置，不再依赖hostfile
NPUS_PER_NODE=8
NNODES=${NNODES:-1}

# Multi-node: extend timeout
if [ "$NNODES" -gt 1 ]; then
    DIST_TIMEOUT="yes"
else
    DIST_TIMEOUT=""
fi
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

echo "Master: $MASTER_ADDR"
echo "Node: $MASTER_ADDR"
echo "Rank: $NODE_RANK"
echo "Nodes: $NNODES"

# 并行配置
MBS=1
GRAD_ACC_STEP=64
TP=1
PP=4
CP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

# 使用传入的配置文件（来自lmsv_rec内部）
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
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
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

# PTA日志目录
LOG_DIR="${LMSV_ROOT}/pta_logs"
mkdir -p ${LOG_DIR}

logfile=$(date +%Y%m%d)_$(date +%H%M%S)_rank${NODE_RANK:-0}
# 多机模式下，loss可能不在NODE_RANK对应的日志中
# 动态选择包含最多loss条目的日志文件
TRAIN_LOG_FILE="${LOG_DIR}/train_${logfile}.log"
if [ ! -f "$TRAIN_LOG_FILE" ] || ! grep -q "loss:" "$TRAIN_LOG_FILE" 2>/dev/null; then
    BEST_LOG=""
    BEST_COUNT=0
    for f in ${LOG_DIR}/train_*.log; do
        if [ -f "$f" ]; then
            count=$(grep -c "loss:" "$f" 2>/dev/null | head -1 || echo 0)
            if [ "$count" -gt "$BEST_COUNT" ]; then
                BEST_COUNT=$count
                BEST_LOG="$f"
            fi
        fi
    done
    if [ -n "$BEST_LOG" ]; then
        TRAIN_LOG_FILE="$BEST_LOG"
    fi
fi

# 进入MindSpeed-MM目录执行
cd ${MINDSPEED_MM_PATH}

echo ""
echo "Starting real training with torchrun..."
echo "PYTHONPATH: ${PYTHONPATH}"
echo "========================================"

# 确保PYTHONPATH传递给torchrun子进程
PYTHONPATH="${MINDSPEED_MM_PATH}:${PYTHONPATH}" torchrun $DISTRIBUTED_ARGS \
    pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee ${LOG_DIR}/train_${logfile}.log 2>&1

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# 多机模式：等待远程节点日志同步到NFS，然后重新选择包含最多loss的日志
if [ "$NNODES" -gt 1 ]; then
    echo "Multi-node mode: waiting for remote logs to sync to NFS..."
    sleep 20
    BEST_LOG=""
    BEST_COUNT=0
    for f in ${LOG_DIR}/train_*.log; do
        if [ -f "$f" ]; then
            count=$(grep -c "loss:" "$f" 2>/dev/null | head -1 || echo 0)
            if [ "$count" -gt "$BEST_COUNT" ]; then
                BEST_COUNT=$count
                BEST_LOG="$f"
            fi
        fi
    done
    if [ -n "$BEST_LOG" ]; then
        TRAIN_LOG_FILE="$BEST_LOG"
        echo "Selected log with most loss entries: $TRAIN_LOG_FILE ($BEST_COUNT losses)"
    else
        echo "WARNING: No log with loss found yet, will retry after delay..."
        sleep 30
        for f in ${LOG_DIR}/train_*.log; do
            if [ -f "$f" ]; then
                count=$(grep -c "loss:" "$f" 2>/dev/null | head -1 || echo 0)
                if [ "$count" -gt "$BEST_COUNT" ]; then
                    BEST_COUNT=$count
                    BEST_LOG="$f"
                fi
            fi
        done
        if [ -n "$BEST_LOG" ]; then
            TRAIN_LOG_FILE="$BEST_LOG"
            echo "Selected log after retry: $TRAIN_LOG_FILE ($BEST_COUNT losses)"
        fi
    fi
fi

chmod 440 ${LOG_DIR}/train_${logfile}.log 2>/dev/null || true
find $SAVE_PATH -type d -exec chmod 750 {} \; 2>/dev/null || true
find $SAVE_PATH -type f -exec chmod 640 {} \; 2>/dev/null || true

# 检查训练是否成功
if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "========================================"
    exit 1
fi

# 检查日志中是否有真正的错误标记（排除Warning）
if grep -Ei "Fatal|RuntimeError|AssertionError" ${TRAIN_LOG_FILE} 2>/dev/null | grep -vi "Warning" > /dev/null; then
    echo ""
    echo "========================================"
    echo "ERROR: Training failed - fatal error detected in logs"
    echo "========================================"
    exit 1
fi

# 检查loss是否存在（多机模式下loss可能在远程节点日志中）
if ! grep -q "loss:" ${TRAIN_LOG_FILE} 2>/dev/null; then
    echo ""
    echo "========================================"
    echo "WARNING: No loss found in local log - may be on remote node"
    echo "========================================"
fi

# 提取关键指标
STEP_TIME=$(grep "elapsed time per iteration" ${TRAIN_LOG_FILE} | awk -F '[:,|]' '{print$5}' | head -n 150 | tail -n 100 | awk '{sum+=$1} END {if (NR>0) print sum/NR; else print "N/A"}')
SAMPLES_PER_SECOND=$(awk -v gbs="$GBS" -v st="$STEP_TIME" 'BEGIN{if(st!="N/A" && st>0) printf("%.3f", gbs*1000/st); else print "N/A"}')

# 提取最终loss和显存 - 严格要求必须有真实数据
LOSS=$(grep "loss:" ${TRAIN_LOG_FILE} | tail -1 | grep -oP 'loss:\s+\K[\d.E+-]+')
# 尝试多种内存格式：NPU memory 或 memory (MB)
MEMORY=$(grep -E "(NPU memory|memory \(MB\))" ${TRAIN_LOG_FILE} | tail -1 | grep -oP 'max allocated:\s*\K[\d.]+')

# 验证提取的指标（多机模式下指标可能在远程节点日志中）
if [ -z "$LOSS" ]; then
    echo ""
    echo "========================================"
    echo "WARNING: Failed to extract loss from training logs"
    echo "========================================"
fi

if [ -z "$MEMORY" ]; then
    echo ""
    echo "========================================"
    echo "WARNING: Failed to extract NPU memory from training logs"
    echo "========================================"
fi

echo ""
echo "========================================"
echo "Training completed successfully"
echo "loss: $LOSS"
echo "NPU memory: $MEMORY MB"
echo "elapsed time per iteration: $STEP_TIME ms"
echo "========================================"
