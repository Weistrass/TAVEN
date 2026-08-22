RUN_NAME=qwen2.5-7b
MODEL_NAME=./checkpoints/$RUN_NAME
MODEL_BASE=llava-v1.5-7b
RESULT_SAVE=CHAIR-eval/results/qwen2.5_7b_visualEnh_greedy

echo "========================="
echo "MODEL: $RUN_NAME"
echo "========================="

CUDA_VISIBLE_DEVICES=0 \
/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python qwen/qwen_model_vqa.py \
--question-file CHAIR-eval/data/chair-500.jsonl \
--image-folder CHAIR-eval/data/chair-500 \
--answers-file $RESULT_SAVE/answer-chair.jsonl \
--beta 0.9 \
--gamma 0.25 \
--alpha 1.15


#--model-base $MODEL_BASE \

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python ./CHAIR-eval/chair.py \
--coco_path /mnt/home/zttian/WenwangHuang/Datasets/mscoco/annotations \
--cache CHAIR-eval/data/chair.pkl \
--cap_file $RESULT_SAVE/answer-chair.jsonl \
--save_path $RESULT_SAVE/eval-chair.json

echo "========================="
echo "MODEL: $RUN_NAME"
echo "========================="