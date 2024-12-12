import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

import wandb
from dataset import CoCoCNDataset
from model import ChineseClipCap
from utils import prepare_wandb, set_seed

set_seed(42)


def train(
    dataset: Dataset,
    dataloader: DataLoader,
    model: ChineseClipCap,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):
    model.train()
    for img, tokens, mask, _ in tqdm(dataloader):
        img, tokens, mask = map(lambda x: x.cuda(), (img, tokens, mask))
        outputs = model(tokens, img, mask)
        logits = outputs.logits[:, dataset.prefix_length - 1:-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               tokens.flatten(),
                               ignore_index=0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        wandb.log({'train/loss': loss.item()})


@torch.no_grad()
def eval(dataset: Dataset, dataloader: DataLoader, model: ChineseClipCap):
    model.eval()
    losses = []
    for img, tokens, mask, _ in tqdm(dataloader):
        img, tokens, mask = map(lambda x: x.cuda(), (img, tokens, mask))
        outputs = model(tokens, img, mask)
        logits = outputs.logits[:, dataset.prefix_length - 1:-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               tokens.flatten(),
                               ignore_index=0,
                               reduction='none')
        losses.extend(loss.tolist())
    eval_loss = sum(losses) / len(losses)
    wandb.log({'eval/loss': round(eval_loss, 2)})
    return eval_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--llm_model',
                        type=str,
                        default='IDEA-CCNL/Wenzhong-GPT2-110M')
    parser.add_argument('--freeze_llm', action='store_true')
    parser.add_argument('--clip_model',
                        type=str,
                        default='thu-ml/zh-clip-vit-roberta-large-patch14')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--only_prefix',
                        dest='only_prefix',
                        action='store_true')
    parser.add_argument('--mapping_type',
                        type=str,
                        default='mlp',
                        help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--normalize_prefix',
                        dest='normalize_prefix',
                        action='store_true')
    args = parser.parse_args()
    prepare_wandb(args.wandb_key, args.project_name, args.run_name)

    prefix_length = args.prefix_length
    # 根据 CLIP 模型出来的 image_features_dim 来确定 prefix_dim
    # 前两种是原生的 CLIP 模型
    if 'ViT' in args.clip_model:
        prefix_dim = 512
    elif 'RN' in args.clip_model:
        prefix_dim = 640
    else:
        prefix_dim = 768
    model = ChineseClipCap(llm_model=args.llm_model,
                           clip_model=args.clip_model,
                           freeze_llm=args.freeze_llm,
                           prefix_length=prefix_length,
                           clip_length=args.prefix_length_clip,
                           prefix_dim=prefix_dim,
                           num_layers=args.num_layers,
                           mapping_type=args.mapping_type)
    model = model.cuda()

    train_dataset = CoCoCNDataset(tokenizer=model.llm.tokenizer,
                                  clip_processor=model.clip.processor,
                                  split='train',
                                  prefix_length=args.prefix_length)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    val_dataset = CoCoCNDataset(tokenizer=model.llm.tokenizer,
                                clip_processor=model.clip.processor,
                                split='val',
                                prefix_length=args.prefix_length)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    epochs = args.epochs
    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps)

    min_loss = 100000
    os.makedirs(f'output/{args.run_name}', exist_ok=True)
    for _ in range(epochs):
        train(train_dataset, train_dataloader, model, optimizer, scheduler)
        eval_loss = eval(val_dataset, val_dataloader, model)
        if eval_loss < min_loss:
            min_loss = eval_loss
            torch.save(
                model.state_dict(),
                os.path.join(f'output/{args.run_name}/pytorch_model.pt'))


if __name__ == '__main__':
    main()
