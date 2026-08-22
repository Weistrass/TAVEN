RUN_NAME=llava-v1.5-7b
MODEL_NAME=./checkpoints/$RUN_NAME
#MODEL_BASE=llava-v1.5-7b
RESULT_SAVE=CHAIR-eval/results/llava_v1.5_7b_greedy

echo "========================="
echo "MODEL: $RUN_NAME"
echo "========================="

CUDA_VISIBLE_DEVICES=0 \
/mnt/home/zttian/WenwangHuang/envs/llava/bin/python -m llava.eval.model_vqa \
--model-path $MODEL_NAME \
--question-file CHAIR-eval/data/chair-500.jsonl \
--image-folder CHAIR-eval/data/chair-500 \
--answers-file $RESULT_SAVE/answer-chair.jsonl \
--temperature 0 \
--conv-mode vicuna_v1 \
--gamma 0.25 \
--alpha 1.15 \
--beta 0.85


#--model-base $MODEL_BASE \

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python CHAIR-eval/chair.py \
--coco_path /mnt/home/zttian/WenwangHuang/Datasets/mscoco/annotations \
--cache CHAIR-eval/data/chair.pkl \
--cap_file $RESULT_SAVE/answer-chair.jsonl \
--save_path $RESULT_SAVE/eval-chair.json

echo "========================="
echo "MODEL: $RUN_NAME"
echo "========================="