#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_GID_INDEX=3
GPUS_PER_NODE=8
# Change for multinode config
#MASTER_ADDR=localhost
#MASTER_PORT=6000
#NNODES=1
#NODE_RANK=0
#WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/root/Megatron-LM/checkpoints/gpt2 #自定义ckp路径
VOCAB_FILE=/root/Megatron-LM/data/gpt2-vocab.json
MERGE_FILE=/root/Megatron-LM/data/gpt2-merges.txt
DATA_PATH=/root/Megatron-LM/data/meg-gpt2-oscar-en-10k_text_document
REMOTE_PATH=/root/megatron-lm/checkpoints/gpt2
LOCAL_PATH=/root/Megatron-LM/checkpoints/gpt2
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr 33.123.210.210 \
    --master_port 23456
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 16 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 16 \
    --global-batch-size 512 \
    --lr 0.00015 \
    --train-iters 1000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 20 \
    --eval-interval 1000 \
    --eval-iters 10
"
#export CUDA_VISIBLE_DIVICES=0,1,2,3
torchrun $DISTRIBUTED_ARGS /root/dp/megatron-lm/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save \
    --save-remote-path $REMOTE_PATH \
    --save-local-path $LOCAL_PATH
