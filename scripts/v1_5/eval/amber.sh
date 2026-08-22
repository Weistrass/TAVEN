WORK_FILE="/mnt/home/zttian/WenwangHuang/opensource/VisionFocus"
QUESTION_FILE="$WORK_FILE/playground/data/eval/amber/data/query/query_generative.json"
OUTPUT_FILE="$WORK_FILE/playground/data/eval/amber/answers"
OUTPUT_NAME="llava-v1.5-7b-greedy-visualEnh-g"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /mnt/home/zttian/WenwangHuang/envs/llava/bin/python -m llava.eval.model_vqa_amber \
    --model-path "$WORK_FILE/checkpoints/llava-v1.5-7b" \
    --question-file "$QUESTION_FILE" \
    --image-folder /mnt/home/zttian/WenwangHuang/Datasets/amber/image \
    --answers-file "$OUTPUT_FILE/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --gamma 0.25 \
    --beta 0.85 \
    --alpha 1.15 \
    --num-chunks "$CHUNKS" \
    --chunk-idx "$IDX" &
done
wait

FINAL_OUTPUT_FILE="$OUTPUT_FILE/$OUTPUT_NAME.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUTPUT_FILE/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >> "$FINAL_OUTPUT_FILE"
    rm "$OUTPUT_FILE/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

/mnt/home/zttian/WenwangHuang/envs/llava/bin/python ./playground/data/eval/amber/eval_amber.py \
    --evaluation_type g \
    --similarity_score 0.8 \
    --metrics "$WORK_FILE/playground/data/eval/amber/data/metrics.txt" \
    --annotation "$WORK_FILE/playground/data/eval/amber/data/annotations.json" \
    --inference_data "$FINAL_OUTPUT_FILE" \
    --safe_words "$WORK_FILE/playground/data/eval/amber/data/safe_words.txt" \
    --word_association "$WORK_FILE/playground/data/eval/amber/data/relation.json"