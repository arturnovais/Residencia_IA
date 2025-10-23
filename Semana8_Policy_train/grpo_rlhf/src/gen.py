def build_generation_kwargs(cfg, tokenizer):
    eot_token = cfg["gen"]["add_eot_token"]
    eot_id = None
    if isinstance(eot_token, str) and eot_token:
        try:
            eot_id = tokenizer.convert_tokens_to_ids(eot_token)
            if not isinstance(eot_id, int) or eot_id < 0:
                eot_id = None
        except Exception:
            eot_id = None

    eos_ids = [tokenizer.eos_token_id]
    if eot_id is not None:
        eos_ids.append(eot_id)

    gkw = {
        "max_new_tokens": int(cfg["gen"]["max_new_tokens"]),
        "do_sample": bool(cfg["gen"]["do_sample"]),
        "temperature": float(cfg["gen"]["temperature"]),
        "top_p": float(cfg["gen"]["top_p"]),
        "repetition_penalty": float(cfg["gen"]["repetition_penalty"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_ids,
    }
    return gkw
