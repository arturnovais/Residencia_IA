from typing import Dict, Tuple, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def _pack(prompt: str, reply: str) -> str:
    return f"User: {prompt}\nAssistant: {reply}"


def _render_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    reply: str,
    use_chat_template: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return _pack(prompt, reply)


def _pairwise_tokenize(
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    use_chat_template: bool,
):
    eos_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

    def _fn(batch: Dict) -> Dict:
        c_txt = [_render_pair(tokenizer, p, c, use_chat_template) for p, c in zip(batch["prompt"], batch["chosen"])]
        r_txt = [_render_pair(tokenizer, p, r, use_chat_template) for p, r in zip(batch["prompt"], batch["rejected"])]

        c_enc = tokenizer(c_txt, max_length=max_len, truncation=True, padding=False)
        r_enc = tokenizer(r_txt, max_length=max_len, truncation=True, padding=False)

        if eos_id is not None:
            for ids in c_enc["input_ids"]:
                if not ids or ids[-1] != eos_id:
                    ids.append(eos_id)
            for ids in r_enc["input_ids"]:
                if not ids or ids[-1] != eos_id:
                    ids.append(eos_id)

        return {
            "input_ids_chosen": c_enc["input_ids"],
            "attention_mask_chosen": c_enc["attention_mask"],
            "input_ids_rejected": r_enc["input_ids"],
            "attention_mask_rejected": r_enc["attention_mask"],
        }

    return _fn


def build_rm_datasets(
    dataset_name: str,
    train_split: str,
    eval_split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_chat_template: bool = True,
) -> Tuple[Dataset, Dataset]:
    ds = load_dataset(dataset_name)
    tr = train_split if train_split in ds else list(ds.keys())[0]
    ev = eval_split if eval_split in ds else [k for k in ds.keys() if k != tr][0]
    print(f"[rm.data] dataset={dataset_name} | train={tr}({len(ds[tr])}) eval={ev}({len(ds[ev])})")

    tok_fn = _pairwise_tokenize(tokenizer, max_length, use_chat_template)
    train_ds = ds[tr].map(tok_fn, batched=True, remove_columns=ds[tr].column_names)
    eval_ds = ds[ev].map(tok_fn, batched=True, remove_columns=ds[ev].column_names)
    return train_ds, eval_ds
