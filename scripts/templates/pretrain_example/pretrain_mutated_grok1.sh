# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="./ckpt_file/"
DATA_PATH="dataset/wiki103-megatron_text_document"
TOKENIZER_MODEL="./assets/runtime/tokenizers/grok1/tokenizer.model"
CKPT_LOAD_DIR=None

TP=1
PP=2
VPP=1
EP=2
CP=1
CP_TYPE=ulysses_cp_algo
NUM_LAYERS=12
SEQ_LEN=2048
MBS=1
GBS=128

#     --gemm-gradient-accumulation-fusion \
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --use-fused-moe-token-permute-and-unpermute \
    --num-experts 8 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 1e-2 \
    --embedding-multiplier-scale 78.38367176906169 \
    --output-multiplier-scale 0.5773502691896257 \
    --input-jitter
"

GPT_ARGS="
    --expert-model-parallel-size ${EP} \
    --use-rotary-position-embeddings \
    --swiglu \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --spec mindspeed_llm.tasks.models.spec.grok_spec layer_spec \
    --post-norm \
    --reuse-fp32-param \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 2048 \
    --ffn-hidden-size 2048 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 2048 \
    --no-gradient-accumulation-fusion \
    --make-vocab-size-divisible-by 1 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --attention-softmax-in-fp32 \
    --vocab-size 131072 \
    --rotary-base 10000 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.0e-5 \
    --train-iters 100 \
    --lr-decay-iters 1280 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2 \
    --clip-grad 1.0 \
    --bf16
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
"


DATA_ARGS="
    --no-shared-storage \
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

torchrun  $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $CKPT_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_grok1_mcore_40b.log
