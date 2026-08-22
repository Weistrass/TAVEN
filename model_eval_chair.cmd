#!/bin/bash

#SBATCH --job-name llava_v1.5_7b_chairEval_greedy 
#SBATCH --mail-type FAIL 
#SBATCH --mail-user 2821024194@qq.com 
#SBATCH --chdir /mnt/home/zttian/WenwangHuang 
#SBATCH --output /mnt/home/zttian/WenwangHuang/output/%j.out 
#SBATCH --error /mnt/home/zttian/WenwangHuang/output/%j.err 
#SBATCH --partition dvlab
#SBATCH --exclude proj77,proj204,proj194,proj198,proj202
#SBATCH --cpus-per-task 12
#SBATCH --gres gpu:1

set -e

cd /mnt/home/zttian/WenwangHuang/opensource/VisionFocus

CUDA_VISIBLE_DEVICES=0 bash ./CHAIR-eval/eval_chair.sh