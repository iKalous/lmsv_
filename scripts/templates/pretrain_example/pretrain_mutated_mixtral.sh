
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

CKPT_SAVE_DIR="./ckpt_file/"
DATA_PATH="dataset/wiki103-megatron_text_document"
TOKENIZER_MODEL="./assets/runtime/tokenizers/mixtral/tokenizer.model"
CKPT_LOAD_DIR=None

TP=1
PP=2
EP=2
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=32

# MOE_ARGS=""
MOE_ARGS="
    --num-experts 4 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --use-fused-moe-token-permute-and-unpermute \
    --use-cp-send-recv-overlap \
"

GPT_ARGS="
    --use-mcore-models  \
    --disable-bias-linear \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 1024  \
    --ffn-hidden-size 2048 \
    --num-attention-heads 4 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 4 \
    --vocab-size 32000 \
    --rotary-base 1000000 \
    --no-masked-softmax-fusion \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-param-gather \
    --make-vocab-size-divisible-by 1 \
    --no-gradient-accumulation-fusion \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE}  \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --lr 1e-5 \
    --train-iters 100 \
    --lr-decay-iters 1280 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2 \
    --clip-grad 1.0 \
    --bf16 \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage \
"

DATA_ARGS="
    --data-path $DATA_PATH  \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 5001 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt_memory.py \
  $MOE_ARGS \
  $GPT_ARGS \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  --distributed-backend nccl \
  | tee logs/train_mixtral_8x7b_ptd.log 
