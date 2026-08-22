import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import types
# use official qwen2.5
#from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

### use ours method
from transformers import AutoProcessor
from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionFlashAttention2, Qwen2_5_VisionTransformerPretrainedModel
from utils import Qwen2_5_VLVisionBlock_foward, Qwen2_5_VLVisionFlashAttention2_forward, Qwen2_5_VisionTransformerPretrainedModel_forward, last_blk_attn_forward, generate_dynamic_qwen25vl
 
Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_foward
Qwen2_5_VLVisionFlashAttention2.forward = Qwen2_5_VLVisionFlashAttention2_forward

Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_forward

Qwen2_5_VLForConditionalGeneration.generate_raw = Qwen2_5_VLForConditionalGeneration.generate
Qwen2_5_VLForConditionalGeneration.generate = generate_dynamic_qwen25vl

# use our method


from qwen_vl_utils import process_vision_info
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

   
    ### our method
    last_blk = model.visual.blocks[-1]
    last_blk.attn.forward = types.MethodType(last_blk_attn_forward, last_blk.attn)
    ### our method

    # default processor
    #processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    # define vision token range
    min_pixels = 256*28*28
    max_pixels = 2048*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["id"]
        image_file = line["image"]
        image_path = os.path.join(args.image_folder, image_file)
        qs = line["query"]

        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": qs},
            ],
        }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        with torch.inference_mode():
            # ours method using
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, enh_para = args.gamma, vision_weight = args.alpha, sup_para = args.beta)

            # baseline using
            #generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0].strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"id": idx,
                                   "response": output_text
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.15)
    parser.add_argument("--use-visaug", action='store_true', default=False)
    parser.add_argument("--dominant_num", type=int, default=576)
    args = parser.parse_args()

    eval_model(args)