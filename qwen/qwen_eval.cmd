#!/bin/bash

#SBATCH --job-name qwen2.5_greedy 
#SBATCH --mail-type FAIL 
#SBATCH --mail-user 2821024194@qq.com 
#SBATCH --chdir /mnt/home/zttian/WenwangHuang 
#SBATCH --output /mnt/home/zttian/WenwangHuang/output/%j.out 
#SBATCH --error /mnt/home/zttian/WenwangHuang/output/%j.err 
#SBATCH --partition dvlab
#SBATCH --exclude proj77,proj204,proj203,proj194,proj198
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:1

set -e

cd /mnt/home/zttian/WenwangHuang/LLaVA

CUDA_VISIBLE_DEVICES=0 bash qwen/scripts/qwen_pope.sh

CUDA_VISIBLE_DEVICES=0 bash qwen/scripts/qwen_chair.sh