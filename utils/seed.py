# utils/seed.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    固定随机种子，确保实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    print(f"[Seed] Fixed random seed = {seed}, deterministic={deterministic}")
