import os, yaml
from typing import Any, Dict

def load_cfg(path: str = None) -> Dict[str, Any]:

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[cfg] usando: {path}")
    return cfg

