

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=2
PP=1
EP=1
CP=2

DATA_PATH="dataset/wiki103-megatron_text_document"
VOCAB_FILE="gpt_dataset/gpt2-vocab.json"
MERGE_FILE="gpt_dataset/gpt2-merges.txt"
TOKENIZER_MODEL="./assets/runtime/tokenizers/baichuan2/tokenizer.model"
TOKENIZER_TYPE="GPT2BPETokenizer"

GPT_ARGS="\
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --num-layers 8 \
    --hidden-size 896 \
    --ffn-hidden-size 2304 \
    --num-attention-heads 16 \
    --vocab-size 102400 \
    --padded-vocab-size 102400 \
    --make-vocab-size-divisible-by 1 \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type GPT2BPETokenizer \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --use-mcore-models \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --reuse-fp32-param \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --train-iters 2000 \
    --lr 2e-06 \
    --lr-decay-style cosine \
    --lr-decay-iters 2000 \
    --min-lr 1e-08 \
    --weight-decay 0.1 \
    --lr-warmup-iters 100 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --bf16 \
    --finetune \
"

MLA_ARGS=""

MOE_ARGS="\
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --use-fused-moe-token-permute-and-unpermute \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 2 \
    --num-experts 64 \
    --moe-router-topk 6 \
    --moe-intermediate-size 1408 \
    --moe-router-load-balancing-type pai_megatron_aux_loss \
    --topk-group 1 \
    --moe-aux-loss-coeff 0.01 \
    --routed-scaling-factor 1.0 \
    --seq-aux \
"

ROPE_ARGS="\
    --rope-scaling-type yarn \
    --rope-scaling-factor 40 \
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-mscale 0.707 \
    --rope-scaling-mscale-all-dim 0.707 \
    --rope-scaling-original-max-position-embeddings 4096 \
"

DATA_ARGS="\
    --data-path $DATA_PATH \
    --split 99,1,0 \
"

OUTPUT_ARGS="\
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 10000 \
    --eval-iters 10 \
    --no-save-optim \
    --no-save-rng \
"

DISTRIBUTED_ARGS="\
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"
