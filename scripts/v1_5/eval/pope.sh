#!/bin/bash
# coco dataset
/mnt/home/zttian/WenwangHuang/envs/llava/bin/python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test_new.jsonl \
    --image-folder /mnt/home/zttian/WenwangHuang/Datasets/mscoco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.6-7b-greedy.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --gamma 0.25 \
    --alpha 1.15 \
    --beta 0.9

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test_new.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.6-7b-greedy.jsonl
