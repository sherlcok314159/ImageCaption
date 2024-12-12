import base64
import json
import os

from tqdm import tqdm

os.makedirs('data/coco/train', exist_ok=True)
os.makedirs('data/coco/val', exist_ok=True)

# 1. 首先从 HF 上下载 jsonl 文件 https://huggingface.co/datasets/MMInstruction/M3IT/tree/main/data/captioning/coco-cn
# 2. 接着将下载的文件进行处理，将图片保存到对应的文件夹中
train_file = './hf_coco_cn/train.jsonl'

train_data = [
    json.loads(line) for line in open(train_file, 'r', encoding='utf-8')
]
for item in tqdm(train_data):
    image_b64 = item['image_base64']
    image_id = item['image_id']
    caption = item['caption']

    image_data = base64.b64decode(image_b64)
    image_path = f'./data/coco/train/COCO_train_{int(image_id):012d}.jpg'
    with open(image_path, 'wb') as f:
        f.write(image_data)

eval_file = './hf_coco_cn/val.jsonl'
eval_data = [
    json.loads(line) for line in open(eval_file, 'r', encoding='utf-8')
]

for item in tqdm(eval_data):
    image_b64 = item['image_base64']
    image_id = item['image_id']
    caption = item['caption']

    image_data = base64.b64decode(image_b64)
    image_path = f'./data/coco/val/COCO_val_{int(image_id):012d}.jpg'
    with open(image_path, 'wb') as f:
        f.write(image_data)
