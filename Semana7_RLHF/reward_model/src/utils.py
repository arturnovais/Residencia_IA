import os, random
import numpy as np
import torch
import yaml

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"[utils] seed aplicada: {seed}")
