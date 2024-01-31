#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
CKPT="henry-vid-cs-bs-0/checkpoint-750/"
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/model_activitynet_qa.py \
    --model-path ./work_dirs/$CKPT \
    --video_dir ./data/ \
    --gt_file_question ./data/test_q.json \
    --gt_file_answers ./data/test_a.json \
    --output_dir ./work_dirs/eval-$CKPT \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &
done