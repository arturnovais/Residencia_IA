from typing import Dict, List, Optional
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    TrainerCallback, TrainingArguments, TrainerState, TrainerControl,
    PreTrainedTokenizerBase
)


class _EvalPairwiseCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of

    def __call__(self, features: List[Dict]):
        c_feats = [{"input_ids": f["input_ids_chosen"], "attention_mask": f["attention_mask_chosen"]} for f in features]
        r_feats = [{"input_ids": f["input_ids_rejected"], "attention_mask": f["attention_mask_rejected"]} for f in features]
        c_pad = self.tok.pad(c_feats, padding=True, pad_to_multiple_of=self.mult, return_tensors="pt")
        r_pad = self.tok.pad(r_feats, padding=True, pad_to_multiple_of=self.mult, return_tensors="pt")
        return {
            "input_ids_chosen": c_pad["input_ids"],
            "attention_mask_chosen": c_pad["attention_mask"],
            "input_ids_rejected": r_pad["input_ids"],
            "attention_mask_rejected": r_pad["attention_mask"],
        }


@torch.no_grad()
def _forward_scores(model, batch, device):
    out_c = model(
        input_ids=batch["input_ids_chosen"].to(device),
        attention_mask=batch["attention_mask_chosen"].to(device),
    )
    out_r = model(
        input_ids=batch["input_ids_rejected"].to(device),
        attention_mask=batch["attention_mask_rejected"].to(device),
    )
    c_logits = out_c.logits.squeeze(-1)
    r_logits = out_r.logits.squeeze(-1)
    return c_logits, r_logits


@torch.no_grad()
def compute_pairwise_metrics_dataloader(
    model,
    eval_loader: DataLoader,
    length_penalty_beta: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    n = wins = ties = 0
    sum_gap = sum_gap_lp = 0.0
    all_margins: List[float] = []
    all_lengths: List[int] = []
    all_scores: List[float] = []

    for batch in eval_loader:
        c_logits, r_logits = _forward_scores(model, batch, device)
        gap = c_logits - r_logits

        wins += (gap > 0).sum().item()
        ties += (gap == 0).sum().item()
        sum_gap += gap.sum().item()
        all_margins.extend(gap.detach().float().cpu().tolist())

        if length_penalty_beta > 0:
            clen = batch["attention_mask_chosen"].sum(dim=1).to(device)
            rlen = batch["attention_mask_rejected"].sum(dim=1).to(device)
            gap_lp = (c_logits - length_penalty_beta * clen) - (r_logits - length_penalty_beta * rlen)
            sum_gap_lp += gap_lp.sum().item()

        all_lengths.extend(
            batch["attention_mask_chosen"].sum(dim=1).tolist()
            + batch["attention_mask_rejected"].sum(dim=1).tolist()
        )
        all_scores.extend(
            c_logits.detach().float().cpu().tolist()
            + r_logits.detach().float().cpu().tolist()
        )

        n += gap.shape[0]

    margins = np.asarray(all_margins, dtype=float)
    lengths = np.asarray(all_lengths, dtype=float)
    scores = np.asarray(all_scores, dtype=float)

    if lengths.size > 1 and scores.size > 1 and np.std(lengths) > 0 and np.std(scores) > 0:
        len_bias_corr = float(np.corrcoef(lengths, scores)[0, 1])
    else:
        len_bias_corr = 0.0

    used = max(1, n - ties)
    out = {
        "pairwise_accuracy": wins / used,
        "violation_rate": 1.0 - (wins / used),
        "ties_ignored": float(ties),
        "reward_gap_mean": sum_gap / max(1, n),
        "margin_p05": float(np.percentile(margins, 5)) if margins.size else 0.0,
        "margin_median": float(np.median(margins)) if margins.size else 0.0,
        "margin_p95": float(np.percentile(margins, 95)) if margins.size else 0.0,
        "len_bias_corr": len_bias_corr,
    }
    if length_penalty_beta > 0:
        out["reward_gap_lenpen_mean"] = sum_gap_lp / max(1, n)
    return out


@torch.no_grad()
def build_preview_table(
    tokenizer: PreTrainedTokenizerBase,
    model,
    dataset,
    rows: int = 8,
    seed: int = 0,
):
    try:
        import wandb
    except Exception:
        return None

    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), k=min(rows, len(dataset)))
    device = next(model.parameters()).device

    data = []
    for i in idxs:
        ex = dataset[i]
        c_ids = torch.tensor(ex["input_ids_chosen"], dtype=torch.long)[None, :]
        r_ids = torch.tensor(ex["input_ids_rejected"], dtype=torch.long)[None, :]
        c_txt = tokenizer.decode(c_ids[0], skip_special_tokens=True)
        r_txt = tokenizer.decode(r_ids[0], skip_special_tokens=True)

        c_logit = model(
            input_ids=c_ids.to(device),
            attention_mask=torch.tensor(ex["attention_mask_chosen"])[None, :].to(device)
        ).logits.item()
        r_logit = model(
            input_ids=r_ids.to(device),
            attention_mask=torch.tensor(ex["attention_mask_rejected"])[None, :].to(device)
        ).logits.item()

        data.append([
            c_logit, r_logit,
            c_logit - r_logit,
            int(sum(ex["attention_mask_chosen"])),
            int(sum(ex["attention_mask_rejected"])),
            c_txt, r_txt
        ])

    return wandb.Table(
        columns=["score_chosen","score_rejected","margin","len_chosen","len_rejected","chosen_text","rejected_text"],
        data=data
    )


class PairwiseMetricsCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        tokenizer: PreTrainedTokenizerBase,
        eval_batch_size: int = 16,
        length_penalty_beta: float = 0.0,
        use_wandb: bool = False,
        num_workers: int = 2,
        pin_memory: bool = True,
        preview_rows: int = 8,
        train_dataset=None,
        train_subset_size: int = 2048,
        train_batch_size: int = 16,
        train_subset_seed: int = 123,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.bs = int(eval_batch_size)
        self.beta = float(length_penalty_beta)
        self.use_wandb = use_wandb
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.preview_rows = preview_rows

        self.train_loader = None
        if train_dataset is not None and train_subset_size > 0:
            rng = random.Random(train_subset_seed)
            idxs = rng.sample(range(len(train_dataset)), k=min(train_subset_size, len(train_dataset)))
            subset = Subset(train_dataset, idxs)
            self.train_loader = DataLoader(
                subset,
                batch_size=int(train_batch_size),
                shuffle=False,
                collate_fn=_EvalPairwiseCollator(self.tokenizer, pad_to_multiple_of=8),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.bs,
            shuffle=False,
            collate_fn=_EvalPairwiseCollator(self.tokenizer, pad_to_multiple_of=8),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        rm_eval = compute_pairwise_metrics_dataloader(
            model=model,
            eval_loader=eval_loader,
            length_penalty_beta=self.beta,
        )

        rm_train = None
        if self.train_loader is not None:
            rm_train = compute_pairwise_metrics_dataloader(
                model=model,
                eval_loader=self.train_loader,
                length_penalty_beta=self.beta,
            )

        def _pfx(d, p): return {f"{p}/{k}": v for k, v in d.items()}
        logdict = _pfx(rm_eval, "eval")
        if rm_train is not None:
            logdict.update(_pfx(rm_train, "train"))

        for k, v in logdict.items():
            print(f"[rm.metrics] {k}={v:.6f}" if isinstance(v, float) else f"[rm.metrics] {k}={v}")

        if metrics is not None:
            metrics.update(logdict)

        if self.use_wandb:
            try:
                import wandb
                wandb.log(logdict, step=state.global_step)
                if self.preview_rows > 0:
                    table = build_preview_table(self.tokenizer, model, self.eval_dataset, rows=self.preview_rows, seed=state.global_step)
                    if table is not None:
                        wandb.log({"eval/sample_pairs": table}, step=state.global_step)
            except Exception:
                pass
