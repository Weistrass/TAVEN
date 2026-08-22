#!/bin/bash

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-greedy-visualEnhLlama.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --gamma 0.25 \
    --beta 0.85 \
    --alpha 1.15

mkdir -p ./playground/data/eval/mm-vet/results

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-greedy-visualEnhLlama.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b-greedy-visualEnhLlama.json

