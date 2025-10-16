import os, yaml, random
import numpy as np
import torch

def load_cfg(path: str | None = None) -> dict:
    path = path or os.environ.get("SFT_CONFIG", "config/train_config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[cfg] loaded: {path}")
    return cfg

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"[seed] set to {seed}")

def enable_tf32():
    torch.set_float32_matmul_precision("high")
