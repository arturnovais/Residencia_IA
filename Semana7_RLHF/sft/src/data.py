from typing import List, Dict
from datasets import load_dataset

def _normalize_messages(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    system_text = None

    for m in msgs:
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue

        if role == "system":
            if system_text is None:
                system_text = content 
            continue

        if role not in ("user", "assistant"):
            continue

        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})

    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    if system_text is not None:
        cleaned.insert(0, {"role": "system", "content": system_text})

    return cleaned

def _ends_with_nonempty_assistant(ex: Dict) -> bool:
    norm = _normalize_messages(ex.get("messages") or [])
    if not norm:
        return False
    if norm[0]["role"] == "system":
        norm = norm[1:]
    return len(norm) >= 2 and norm[-1]["role"] == "assistant" and norm[-1]["content"].strip() != ""

def build_datasets(cfg, tokenizer):
    ds_all = load_dataset(cfg["data"]["dataset_name"])
    tr, ev = cfg["data"]["train_split"], cfg["data"]["eval_split"]
    ds_train = ds_all[tr].filter(_ends_with_nonempty_assistant)
    ds_eval  = ds_all[ev].filter(_ends_with_nonempty_assistant)

    num_proc = int(cfg["data"].get("num_proc", 1))
    max_len  = int(cfg["data"]["max_seq_length"])

    def _fmt(batch):
        texts = []
        for msgs in batch["messages"]:
            norm = _normalize_messages(msgs)
            texts.append(
                tokenizer.apply_chat_template(
                    norm, tokenize=False, add_generation_prompt=False
                )
            )
        return {"text": texts}

    ds_train = ds_train.map(_fmt, batched=True, remove_columns=ds_train.column_names,
                            num_proc=num_proc, load_from_cache_file=False)
    ds_eval  = ds_eval.map(_fmt, batched=True, remove_columns=ds_eval.column_names,
                           num_proc=num_proc, load_from_cache_file=False)

    def _tok(batch):
        enc = tokenizer(batch["text"], add_special_tokens=False, return_attention_mask=True)
        out_ids, out_attn = [], []
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            if len(ids) > max_len:
                ids  = ids[-max_len:]
                attn = attn[-max_len:]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            if len(ids) < max_len:
                pad = max_len - len(ids)
                ids  = ids + [pad_id] * pad
                attn = attn + [0] * pad
            out_ids.append(ids)
            out_attn.append(attn)
        return {"input_ids": out_ids, "attention_mask": out_attn}

    ds_train = ds_train.map(_tok, batched=True, remove_columns=["text"],
                            num_proc=num_proc, load_from_cache_file=False)
    ds_eval  = ds_eval.map(_tok, batched=True, remove_columns=["text"],
                           num_proc=num_proc, load_from_cache_file=False)

    return ds_train, ds_eval
