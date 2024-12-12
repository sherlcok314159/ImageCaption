import json
import os
from collections import defaultdict
from typing import Union

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers import AutoTokenizer

from zhclip.models import ZhCLIPProcessor

Tensor = torch.Tensor
RISIZE_SHAPE = (256, 256)


def load_jsonl(file_path: str) -> list[dict[str]]:
    return [
        json.loads(line) for line in open(file_path, 'r', encoding='utf-8')
    ]


class CoCoCNDataset(Dataset):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        clip_processor: Union[ZhCLIPProcessor, Compose],
        split: str,
        prefix_length: int,
    ):
        print(f'对 {split} 集进行预处理... ')
        if split == 'train':
            dataset = load_jsonl('hf_coco_cn/train.jsonl')
        else:
            dataset = load_jsonl('hf_coco_cn/val.jsonl')

        caption2tokens = {}
        labels_dict = defaultdict(list)
        for item in tqdm(dataset):
            img_id = item['image_id']
            caption = item['caption']
            tokens = tokenizer.encode(caption, return_tensors='pt').squeeze(0)
            caption = ''.join(caption)
            labels_dict[img_id].append((tokens, caption, len(caption)))
            caption2tokens[caption] = tokens

        labels = []
        for img_id, captions in tqdm(labels_dict.items()):
            for tokens, caption, caplen in captions:
                labels.append((img_id, tokens, caption, caplen))

        self.labels = labels
        self.labels_dict = labels_dict
        self.split = split
        self.caption2tokens = caption2tokens
        self.prefix_length = prefix_length
        self.clip_processor = clip_processor

        all_len = torch.tensor(
            [len(self.labels[i][2]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10),
                               int(all_len.max()))

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor, int]:
        img_id, tokens = self.labels[i][:2]
        img = self.preprocess(img_id, self.split)
        # 如果是灰度图像，复制三份
        if img.shape[0] == 1:
            img = torch.cat([img] * 3, dim=0)

        assert img.shape[0] == 3, f'img.shape == {img.shape}'
        tokens, mask = self.pad_tokens(tokens)
        return img, tokens, mask, img_id

    def __len__(self) -> int:
        return len(self.labels)

    def pad_tokens(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        # 给 pad 部分添加 mask
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        # 给 prefix 部分添加 mask
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def preprocess(self, img_id: int, split: str) -> Tensor:
        path = f'./data/coco/{split}/COCO_{split}_{int(img_id):012d}.jpg'
        resized_path = f'./data/coco/{split}/resized_COCO_{split}_{int(img_id):012d}.jpg'

        if not os.path.exists(resized_path):
            img = cv2.imread(path)
            img = cv2.resize(img, RISIZE_SHAPE)
            cv2.imwrite(resized_path, img)

        img_pil = Image.open(resized_path)

        if isinstance(self.clip_processor, Compose):
            img_tensor = self.clip_processor(img_pil).squeeze(0)
        else:
            img_tensor = self.clip_processor(
                images=img_pil, return_tensors='pt')['pixel_values'].squeeze(0)
        return img_tensor
