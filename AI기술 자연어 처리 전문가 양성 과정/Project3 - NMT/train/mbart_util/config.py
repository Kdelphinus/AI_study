import torch
import numpy as np
import random


class Config:
    device = "cuda"
    seed = 42
    epoch = 1
    max_train_steps = 10000
    batch_size = 4
    gradient_accumulation = 4
    max_seq_len = 1024
    learning_rate = 3e-5
    warmup_steps = 100
    grad_clip_norm = 1.0
    optimizer_weight_decay = 0.0
    save_path = "logs"
    model_save_dir = f"{save_path}"


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
