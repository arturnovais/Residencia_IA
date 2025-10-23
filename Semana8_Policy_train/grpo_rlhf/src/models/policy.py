import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from safetensors.torch import load as safe_load_buffer

def load_policy(policy_model_path: str, dtype=None):

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    weights_path = os.path.join(policy_model_path, "model.safetensors")
    with open(weights_path, "rb") as f:
        sd = safe_load_buffer(f.read())
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    hf_cfg = AutoConfig.from_pretrained(policy_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device="cuda", dtype=None if dtype is torch.float32 else dtype)

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.use_cache = True

    tok = AutoTokenizer.from_pretrained(policy_model_path, use_fast=True, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    return model, tok
