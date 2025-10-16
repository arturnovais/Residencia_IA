import os
import torch
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from src.callbacks.qual_eval import QualEvalCallback

ANCHOR_TEXT = "<start_of_turn>model\n"
END_TEXT    = "<end_of_turn>"

def make_multiturn_collator(tokenizer, include_end_tokens: bool = True):
    base = DataCollatorForCompletionOnlyLM(response_template=ANCHOR_TEXT, tokenizer=tokenizer)
    anchor_ids = tokenizer(ANCHOR_TEXT, add_special_tokens=False)["input_ids"]
    end_ids    = tokenizer(END_TEXT,    add_special_tokens=False)["input_ids"]
    ka, ke = len(anchor_ids), len(end_ids)

    def _find_all(x: torch.Tensor, pat: torch.Tensor):
        out, L, K = [], x.numel(), pat.numel()
        if K == 0 or K > L:
            return out
        for i in range(0, L - K + 1):
            if torch.equal(x[i:i+K], pat):
                out.append(i)
        return out

    def collate(examples):
        batch = base(examples)
        x = batch["input_ids"]; labs = batch["labels"]; attn = batch.get("attention_mask", None)
        a = torch.tensor(anchor_ids, device=x.device); e = torch.tensor(end_ids, device=x.device)

        for i in range(x.size(0)):
            ids, lab = x[i], labs[i]
            lab.fill_(-100)

            seq_end = ids.size(0)
            if attn is not None:
                on = torch.nonzero(attn[i] == 1, as_tuple=False)
                if on.numel(): seq_end = int(on[-1].item()) + 1

            for s in _find_all(ids[:seq_end], a):
                start_out = s + ka
                cut = -1
                for j in range(start_out, seq_end - ke + 1):
                    if torch.equal(ids[j:j+ke], e):
                        cut = j; break
                end_out = (cut + (ke if include_end_tokens else 0)) if cut >= 0 else seq_end
                if end_out > start_out:
                    lab[start_out:end_out] = ids[start_out:end_out]

            if attn is not None:
                lab[attn[i] == 0] = -100

        batch["labels"] = labs
        return batch

    return collate

def build_sft_trainer(cfg: dict, model, tokenizer, train_ds, eval_ds) -> SFTTrainer:
    args = SFTConfig(
        output_dir=cfg["run"]["output_dir"],
        num_train_epochs=float(cfg["train"]["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["train"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        max_grad_norm=float(cfg["train"].get("max_grad_norm", 1.0)),
        logging_steps=int(cfg["train"]["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(cfg["train"]["eval_steps"]),
        save_strategy=str(cfg["train"]["save_strategy"]),
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        lr_scheduler_type="cosine",
        bf16=bool(cfg["model"].get("bf16", True)),
        bf16_full_eval=True,
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        packing=bool(cfg["data"].get("packing", False)),
        max_seq_length=int(cfg["data"]["max_seq_length"]),
        optim="adamw_torch_fused",
        group_by_length=False,
        dataloader_num_workers=int(cfg["train"].get("num_workers", 0)),
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        report_to=[cfg["run"]["report_to"]] if cfg["run"]["report_to"] != "none" else [],
    )

    collator = make_multiturn_collator(tokenizer, include_end_tokens=True)

    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        data_collator=collator,
    )

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    try:
        model = torch.compile(model, mode="max-autotune")
        trainer.model = model
    except Exception:
        pass

    is_main = (int(os.environ.get("RANK", "0")) == 0)
    if cfg["run"].get("enable_qual_eval", True):
        prompts = cfg["run"].get("qual_eval_prompts")  
        trainer.add_callback(
            QualEvalCallback(
                tokenizer,
                is_main=is_main,
                prompts=prompts,
                max_new_tokens=int(cfg["data"].get("gen_max_new_tokens", 192)),
            )
        )

    return trainer
