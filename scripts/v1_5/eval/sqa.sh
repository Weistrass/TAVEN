#!/bin/bash

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /mnt/home/zttian/WenwangHuang/Datasets/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-greedy-visualEnhLlama.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --gamma 0.25 \
    --beta 0.85 \
    --alpha 1.15

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-greedy-visualEnhLlama.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-greedy-visualEnhLlama_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-greedy-visualEnhLlama_result.json
