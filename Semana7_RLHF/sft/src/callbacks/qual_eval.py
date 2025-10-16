from __future__ import annotations
from typing import List, Dict, Optional
import os
import torch
from transformers import TrainerCallback, StoppingCriteria, StoppingCriteriaList

END_TEXT = "<end_of_turn>"

class _StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids, prompt_len):
        self.stop_ids = torch.tensor(stop_ids)
        self.k = len(stop_ids)
        self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0][self.prompt_len:]
        if seq.numel() < self.k:
            return False
        return torch.equal(seq[-self.k:], self.stop_ids.to(seq.device))

def _make_inputs(tok, model, messages):
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(enc, torch.Tensor):
        enc = {"input_ids": enc}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return {k: v.to(model.device) for k, v in enc.items()}

def generate_once(model, tok, messages: List[Dict[str, str]], *, max_new_tokens=192, temperature=0.7, top_p=0.9):
    model.eval()
    with torch.inference_mode():
        enc = _make_inputs(tok, model, messages)
        prompt_len = enc["input_ids"].shape[1]

        old_cache = getattr(model.config, "use_cache", True)
        try:
            model.config.use_cache = True
        except Exception:
            pass

        end_ids = tok(END_TEXT, add_special_tokens=False)["input_ids"]
        eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
        stopping = None
        if len(end_ids) == 1:
            eos_ids.append(end_ids[0])
        elif len(end_ids) > 1:
            stopping = StoppingCriteriaList([_StopOnSubsequence(end_ids, prompt_len)])

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
            eos_token_id=eos_ids if eos_ids else None,
            stopping_criteria=stopping,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

        try:
            model.config.use_cache = old_cache
        except Exception:
            pass

    gen = out[0][prompt_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text.split(END_TEXT)[0].strip()

class QualEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        *,
        is_main: bool,
        prompts: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 192,
    ):
        self.tok = tokenizer
        self.is_main = is_main
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.prompts = prompts or [
            {"name": "p0_en_greet", "user": "hi, how are you?"},
            {"name": "p1_en_rlhf",  "user": "Explain RLHF in two concise sentences."},
            {"name": "p2_pt_trad",  "user": 'translate "ball" to Portuguese'},
            {"name": "p3_pt_dicas", "user": "what is 2 + 2?"},
            {"name": "p4_en_turing","user": "Who was Alan Turing? Answer in one sentence."},
        ]
        try:
            import wandb  # noqa: F401
            self._has_wandb = True
        except Exception:
            self._has_wandb = False

    def _messages(self, user_text: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_text},
        ]

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.is_main:
            return
        model = kwargs.get("model")
        if model is None:
            return

        rows = []
        print("\n[qual-eval] ====== qualitative eval outputs ======")
        for item in self.prompts:
            try:
                ans = generate_once(
                    model, self.tok, self._messages(item["user"]),
                    max_new_tokens=self.max_new_tokens
                )
            except Exception as e:
                ans = f"[generation error] {e}"
            print(f"\n--- {item['name']} ---")
            print(f"USER: {item['user']}")
            print(f"ASSISTANT: {ans}")
            rows.append([item["name"], item["user"], ans])

        if self._has_wandb and os.environ.get("WANDB_MODE", "") != "disabled":
            try:
                import wandb
                table = wandb.Table(data=rows, columns=["name", "user", "assistant"])
                wandb.log({"qual_eval_table": table}, step=state.global_step)
            except Exception:
                pass
        print("[qual-eval] =======================================\n")
