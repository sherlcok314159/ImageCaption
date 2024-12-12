import random
import os

import numpy as np
import torch

import wandb


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_wandb(wandb_key: str, project_name: str, run_name: str):
    os.environ['WANDB_API_KEY'] = wandb_key
    wandb.init(project=project_name, name=run_name)
