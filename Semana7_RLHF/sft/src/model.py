import torch, torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def build_tokenizer(model_name: str, tokenizer_name: str | None, max_len: int):
    tok_id = tokenizer_name or model_name
    tok = AutoTokenizer.from_pretrained(
        tok_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = max_len
    print(f"[tok] {tok_id} | pad={tok.pad_token_id} | chat_template={'OK' if getattr(tok,'chat_template',None) else 'NONE'}")
    return tok

def build_model(model_name: str, bf16: bool, lora_cfg: dict | None, tokenizer=None):
    dtype = (
        torch.bfloat16 if (bf16 and torch.cuda.is_available())
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    cfg = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    

    try:
        model.config.attn_implementation = "flash_attention_2"
        print("[model] flash_attention_2 enabled")
    except Exception:
        print("[model] flash_attention_2 not available")

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if tokenizer is not None:
        nv = len(tokenizer)
        ov = model.get_input_embeddings().weight.size(0)
        if nv != ov:
            print(f"[model] resize_token_embeddings {ov}->{nv}")
            model.resize_token_embeddings(nv)
            
    return model