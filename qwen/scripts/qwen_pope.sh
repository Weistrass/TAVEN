
# coco pope 9k
/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python qwen/qwen_model_loader_vqa.py \
    --question-file ./playground/data/eval/pope/llava_pope_test_new.jsonl \
    --image-folder /mnt/home/zttian/WenwangHuang/Datasets/mscoco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/qwen2.5-7b-greedy-visualEnh.jsonl \
    --beta 0.9 \
    --gamma 0.25 \
    --alpha 1.15

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test_new.jsonl \
    --result-file ./playground/data/eval/pope/answers/qwen2.5-7b-greedy-visualEnh.jsonl