import inspect
from trl import RewardTrainer, RewardConfig
from transformers import PreTrainedTokenizerBase


def _has_arg(cls, name: str) -> bool:
    return name in inspect.signature(cls.__init__).parameters


class PairwiseDynamicCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int = 8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of

    def __call__(self, features):
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


def build_trainer(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    output_dir,
    num_train_epochs,
    learning_rate,
    warmup_ratio,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    logging_steps,
    save_total_limit,
    bf16,
    report_to="none",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
):
    cfg = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "save_total_limit": save_total_limit,
        "bf16": bf16,
        "report_to": [report_to] if report_to != "none" else [],
        "remove_unused_columns": False,
        "disable_dropout": True,
    }

    if _has_arg(RewardConfig, "evaluation_strategy"):
        cfg["evaluation_strategy"] = "steps"
    if _has_arg(RewardConfig, "num_print_samples"):
        cfg["num_print_samples"] = 0
    elif _has_arg(RewardConfig, "eval_strategy"):
        cfg["eval_strategy"] = "steps"
    if _has_arg(RewardConfig, "eval_steps"):
        cfg["eval_steps"] = eval_steps
    if _has_arg(RewardConfig, "save_strategy"):
        cfg["save_strategy"] = save_strategy
    if save_strategy == "steps" and _has_arg(RewardConfig, "save_steps"):
        cfg["save_steps"] = save_steps
    if _has_arg(RewardConfig, "logging_strategy"):
        cfg["logging_strategy"] = "steps"

    args = RewardConfig(**cfg)

    return RewardTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PairwiseDynamicCollator(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
    )
