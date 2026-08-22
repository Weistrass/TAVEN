OUTPUT_NAME="qwen2_5_7b_greedy"
#OUTPUT_NAME="qwen2_5_7b_visualEnhv23_greedy"

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python qwen/qwen_model_loader_vqa.py \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file "./playground/data/eval/MME/answers/${OUTPUT_NAME}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --beta 0.9 \
    --gamma 0.25 \
    --dominant_num 576 \
    --alpha 1.15
    # --use-visaug

cd ./playground/data/eval/MME

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python convert_answer_to_mme.py --experiment "$OUTPUT_NAME"

cd eval_tool

/mnt/home/zttian/WenwangHuang/envs/qwen2.5/bin/python calculation.py --results_dir "answers/${OUTPUT_NAME}"