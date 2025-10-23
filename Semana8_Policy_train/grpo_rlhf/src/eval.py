import re
import torch
from typing import List, Dict, Optional, Tuple
from transformers import TrainerCallback

try:
    import wandb
except Exception:
    wandb = None


def _clean_completion(text: str, prompt: str | None = None) -> str:
    t = text.strip().replace("<end_of_turn>", "").strip()
    lines = t.splitlines()
    while lines and re.match(r"^(user|assistant|model)\s*:?$", lines[0].strip(), flags=re.I):
        lines.pop(0)
    if lines:
        lines[0] = re.sub(r"^(assistant|model)\s*:\s*", "", lines[0].strip(), flags=re.I)
    t = "\n".join(lines).strip()

    if prompt:
        pref_cands = [
            prompt.strip(),
            f"user\n{prompt.strip()}",
            f"User: {prompt.strip()}",
            f"### User:\n{prompt.strip()}",
        ]
        for pref in pref_cands:
            if t.startswith(pref):
                t = t[len(pref):].lstrip()
                break
    return t


@torch.no_grad()
def simple_eval(
    model,
    tokenizer,
    rm_model,
    rm_tokenizer,
    dataset,
    n_samples: int = 512,
    batch_size: int = 8,
    generation_kwargs: Optional[dict] = None,
    examples_k: int = 4,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    device = next(model.parameters()).device
    model.eval()

    N = min(n_samples, len(dataset))
    prompts = [dataset[i]["prompt"] for i in range(N)]

    def _pad(tensors: List[torch.Tensor], pad_id: int, side: str = "left"):
        maxlen = max(t.size(0) for t in tensors)
        out_ids = torch.full((len(tensors), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(tensors), maxlen), dtype=torch.long)
        for i, t in enumerate(tensors):
            L = t.size(0)
            if side == "left":
                out_ids[i, -L:] = t
                attn[i, -L:] = 1
            else:
                out_ids[i, :L] = t
                attn[i, :L] = 1
        return out_ids, attn

    completions: List[str] = []
    gen_lens: List[int] = []

    gkw = dict(generation_kwargs or {})
    gkw.setdefault("pad_token_id", tokenizer.pad_token_id)
    gkw.setdefault("eos_token_id", tokenizer.eos_token_id)
    pad_side = getattr(tokenizer, "padding_side", "left")

    for i in range(0, N, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        toks: List[torch.Tensor] = []
        for p in batch_prompts:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            toks.append(ids)

        input_ids, attention_mask = _pad(toks, tokenizer.pad_token_id, side=pad_side)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gkw,
        )

        input_lens = attention_mask.sum(dim=1)
        for j in range(out.size(0)):
            L = int(input_lens[j].item())
            gen_ids = out[j, L:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            text = _clean_completion(raw, prompt=batch_prompts[j])
            completions.append(text)
            gen_lens.append(int(gen_ids.size(0)))

    rewards: List[float] = []
    rm_device = next(rm_model.parameters()).device
    for i in range(0, N, batch_size):
        texts = [
            f"User: {p}\nAssistant: {c}"
            for p, c in zip(prompts[i:i + batch_size], completions[i:i + batch_size])
        ]
        enc = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(rm_device) for k, v in enc.items()}
        scores = rm_model(**enc).logits.squeeze(-1).float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        rewards.extend(scores)

    import statistics as stats
    mean_reward = float(stats.fmean(rewards)) if rewards else float("nan")
    std_reward  = float(stats.pstdev(rewards)) if len(rewards) > 1 else 0.0
    mean_len    = float(stats.fmean(gen_lens)) if gen_lens else float("nan")

    metrics = {
        "eval/reward_mean": mean_reward,
        "eval/reward_std": std_reward,
        "eval/len_mean": mean_len,
        "eval/n": float(N),
    }

    K = min(examples_k, N)
    examples = []
    for i in range(K):
        examples.append({
            "prompt": prompts[i],
            "completion": completions[i],
            "reward": f"{rewards[i]:.4f}",
        })

    return metrics, examples


class PeriodicEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_every_steps: int,
        rm_model,
        rm_tokenizer,
        test_dataset,
        tokenizer,
        n_samples: int = 512,
        batch_size: int = 8,
        generation_kwargs: Optional[dict] = None,
        examples_k: int = 4,
    ):
        self.eval_every = int(eval_every_steps)
        self.rm_model = rm_model
        self.rm_tok = rm_tokenizer
        self.ds = test_dataset
        self.tok = tokenizer
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.gkw = dict(generation_kwargs or {})
        self.examples_k = int(examples_k)
        self.trainer = None
        self._wandb_ok = False

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb as wb
        except Exception:
            wb = None
        self._wandb_ok = bool(wb and getattr(wb, "run", None))
        if self._wandb_ok:
            try:
                wb.define_metric("train/global_step")
                wb.define_metric("eval/*", step_metric="train/global_step")
            except Exception:
                pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and (state.global_step % self.eval_every == 0):
            model = kwargs["model"]
            metrics, examples = simple_eval(
                model=model,
                tokenizer=self.tok,
                rm_model=self.rm_model,
                rm_tokenizer=self.rm_tok,
                dataset=self.ds,
                n_samples=self.n_samples,
                batch_size=self.batch_size,
                generation_kwargs=self.gkw,
                examples_k=self.examples_k,
            )

            eval_metrics = {}
            for k, v in metrics.items():
                if k == "step":
                    continue
                k2 = k if k.startswith("eval/") else f"eval/{k}"
                eval_metrics[k2] = v

            if self._wandb_ok:
                import wandb as wb
                payload = {"train/global_step": int(state.global_step), **eval_metrics}
                wb.log(payload, commit=True)

                table = wb.Table(columns=["step", "prompt", "answer", "reward"])
                for ex in examples:
                    table.add_data(int(state.global_step), ex["prompt"], ex["completion"], ex["reward"])
                wb.log({"train/global_step": int(state.global_step),
                        "eval/examples": table}, commit=False)
            else:
                print("[eval]", {"step": state.global_step, **eval_metrics})

            print("\n" + "=" * 80)
            print(f"[EVAL @ step {state.global_step}] exemplos qualitativos")
            print("-" * 80)
            for idx, ex in enumerate(examples, 1):
                print(f"#{idx}  reward={ex['reward']}")
                print("┌ PROMPT:")
                print("│ " + ex["prompt"].replace("\n", "\n│ "))
                print("├ ANSWER (modelo):")
                print("│ " + ex["completion"].replace("\n", "\n│ "))
                print("└" + "-" * 76)
            print("=" * 80 + "\n")
