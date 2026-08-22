import os
import json

# Collect images from MSCOCO

anno_file = './CHAIR-eval/data/chair-500.jsonl'
#anno = json.load(open(anno_file)) #原始代码，但针对的对象是.json文件，对于.jsonl文件需要额外处理
anno = []
with open(anno_file, 'r') as f:
    for line in f:
        anno.append(json.loads(line.strip()))

image_folder = '/mnt/home/zttian/WenwangHuang/Datasets/mscoco/val2014'
output_folder = './CHAIR-eval/data/chair-500'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for each in anno:
    original_image = each['image']
    image = "COCO_val2014_"+original_image
    os.symlink(os.path.join(image_folder, image), os.path.join(output_folder, original_image))