from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def load_rm_model_and_tokenizer_llm(
    model_name: str,
    bf16: bool,
    max_length: int,
    use_lora: bool,
    qlora_4bit: bool,  
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tok.pad_token is None:
        if tok.eos_token is None:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = max_length
    tok.truncation_side = "left"

    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
        torch_dtype=dtype,
        device_map="auto",
    )

    if tok.pad_token_id is not None and tok.pad_token_id >= model.config.vocab_size:
        model.resize_token_embeddings(len(tok))

    model.config.pad_token_id = tok.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id

    if use_lora:
        peft_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["score"]
        )
        model = get_peft_model(model, peft_cfg)

    if hasattr(model, "score") and isinstance(model.score, nn.Linear):
        nn.init.normal_(model.score.weight, mean=0.0, std=0.02)
        if model.score.bias is not None:
            nn.init.zeros_(model.score.bias)

    tr, tot = _count_params(model)
    print(f"[rm.model] {model_name} | bf16={bf16} | LoRA={use_lora} (r={r}, alpha={lora_alpha}, drop={lora_dropout})")
    print(f"[rm.model] params: trainable={tr:,} / total={tot:,}")
    print(f"[rm.model] max_len={max_length} | pad_token_id={tok.pad_token_id} | padding_side={tok.padding_side}")
    print(f"[rm.model] config.pad_token_id={getattr(model.config, 'pad_token_id', None)}")

    return model, tok
