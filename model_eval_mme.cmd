#!/bin/bash

#SBATCH --job-name llava_v1.6_7b_mmeEval 
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
echo "开始执行MME评估脚本"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
echo "开始执行CHAIR评估脚本"
CUDA_VISIBLE_DEVICES=0 bash ./CHAIR-eval/eval_chair.sh
echo "开始执行ScienceQA评估脚本"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh
echo "开始执行MMVET评估脚本"
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
