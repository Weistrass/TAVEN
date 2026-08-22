#!/bin/bash

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python qwen/qwen_model_vqa.py \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/qwen2.5-7b-greedy.jsonl \
    --gamma 0.25 \
    --alpha 1.15 \
    --beta 0.9

mkdir -p ./playground/data/eval/mm-vet/results

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/qwen2.5-7b-greedy.jsonl \
    --dst ./playground/data/eval/mm-vet/results/qwen2.5-7b-greedy.json
