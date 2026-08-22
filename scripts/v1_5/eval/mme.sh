#!/bin/bash

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b-greedy-visualEnhLlama.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --gamma 0.25 \
    --beta 0.85 \
    --alpha 1.15

cd ./playground/data/eval/MME

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python convert_answer_to_mme.py --experiment llava-v1.5-7b-greedy-visualEnhLlama

cd eval_tool

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python calculation.py --results_dir answers/llava-v1.5-7b-greedy-visualEnhLlama
