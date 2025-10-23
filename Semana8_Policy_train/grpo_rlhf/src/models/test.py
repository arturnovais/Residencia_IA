import torch

def quick_smoke_test(policy_model, policy_tok, rm_model, rm_tok, generation_kwargs: dict):
    prompt = "hi, how are you?"
    msgs = [{"role": "user", "content": prompt}]

    enc = policy_tok.apply_chat_template(
        msgs, tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).to(policy_model.device)
    enc = {"input_ids": enc, "attention_mask": torch.ones_like(enc)}
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = policy_model.generate(**enc, **generation_kwargs)

    gen_text = policy_tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
    if "<end_of_turn>" in gen_text:
        gen_text = gen_text.split("<end_of_turn>")[0].strip()

    bad_text = "I'm not good, i hope you're bad"

    from .rm import rm_score_single
    r_gen = rm_score_single(rm_model, rm_tok, prompt, gen_text, max_length=2048)
    r_bad = rm_score_single(rm_model, rm_tok, prompt, bad_text, max_length=2048)

    print("===== PROMPT =====")
    print(prompt)
    print("\n===== RESPOSTA (policy) =====")
    print(gen_text)
    print("\n===== BASELINE (ruim) =====")
    print(bad_text)
    print("\n===== REWARDS =====")
    print(f"policy:   {r_gen:.4f}")
    print(f"baseline: {r_bad:.4f}")
    melhor = "policy" if r_gen >= r_bad else "baseline"
    print(f"\nMelhor: {melhor}  (↑ maior é melhor)\n")
