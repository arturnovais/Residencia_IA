import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def load_rm(rm_base_name: str, rm_adapter_path: str, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    rm_tok = AutoTokenizer.from_pretrained(rm_base_name, use_fast=True, trust_remote_code=True)
    rm_tok.pad_token = rm_tok.pad_token or rm_tok.eos_token
    rm_tok.padding_side = "right"

    rm_base = AutoModelForSequenceClassification.from_pretrained(
        rm_base_name,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    rm = PeftModel.from_pretrained(rm_base, rm_adapter_path)
    rm.config.pad_token_id = rm_tok.pad_token_id
    rm.eval()

    return rm, rm_tok

@torch.no_grad()
def rm_score_single(rm_model, rm_tok, prompt: str, answer: str, max_length: int = 2048) -> float:
    text = f"User: {prompt}\nAssistant: {answer}"
    enc = rm_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    dev = next(rm_model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    return float(rm_model(**enc).logits.squeeze(-1).item())

@torch.no_grad()
def make_reward_fn(rm_model, rm_tok, max_length: int = 2048, batch_size: int = 8):
    dev = next(rm_model.parameters()).device

    def reward_fn(samples=None, prompts=None, completions=None, **kwargs):
        if samples is None and completions is not None:
            samples = completions
        if prompts is None:
            prompts = kwargs.get("prompts", [""] * len(samples))

        scores: List[float] = []
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            texts = [f"User: {p}\nAssistant: {s}" for p, s in zip(batch_prompts, batch_samples)]
            enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = rm_model(**enc).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(out, float):
                out = [out]
            scores.extend(out)
        return scores

    return reward_fn
