from datasets import  Dataset, load_dataset
from transformers import AutoTokenizer


def build_dataset(dataset_name: str,
                  split: str,) -> Dataset:
    
    ds = load_dataset(dataset_name, split=split)
    ds = ds.remove_columns([c for c in ds.column_names if c != "prompt"])
        
    #ds = ds.rename_column("prompt", "query") SÓ NO PPO
    return ds









