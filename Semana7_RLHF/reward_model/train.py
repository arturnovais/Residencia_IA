import os
import argparse
from src.utils import load_config, set_global_seed
from src.model import load_rm_model_and_tokenizer_llm
from src.data import build_rm_datasets
from src.trainer import build_trainer
from src.metrics_callback import PairwiseMetricsCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/rm_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[cfg] usando: {args.config}")
    cfg = load_config(args.config)

    rank = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)

    want_wandb = (cfg["run"].get("report_to", "none") == "wandb")
    use_wandb = want_wandb and is_main
    if want_wandb and not is_main:
        os.environ["WANDB_MODE"] = "disabled"
    if use_wandb:
        import wandb
        w = cfg["run"].get("wandb", {})
        wandb.init(
            project=w.get("project", "rm"),
            entity=w.get("entity"),
            name=w.get("run_name"),
            tags=w.get("tags", []),
            config=cfg,
        )
        os.environ.setdefault("WANDB_DIR", os.path.join(cfg["run"]["output_dir"], "wandb"))

    set_global_seed(cfg["run"].get("seed", 42))

    model, tok = load_rm_model_and_tokenizer_llm(
        model_name=cfg["model"]["name"],
        bf16=bool(cfg["model"].get("bf16", True)),
        max_length=int(cfg["model"].get("max_length", 512)),
        use_lora=bool(cfg["peft"].get("use_lora", True)),
        qlora_4bit=False,
        r=int(cfg["peft"].get("r", 16)),
        lora_alpha=int(cfg["peft"].get("lora_alpha", 32)),
        lora_dropout=float(cfg["peft"].get("lora_dropout", 0.05)),
        target_modules=cfg["peft"].get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    train_ds, eval_ds = build_rm_datasets(
        dataset_name=cfg["data"]["dataset_name"],
        train_split=cfg["data"]["train_split"],
        eval_split=cfg["data"]["eval_split"],
        tokenizer=tok,
        max_length=int(cfg["model"]["max_length"]),
        use_chat_template=bool(cfg["data"].get("use_chat_template", True)),
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tok,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=cfg["run"]["output_dir"],
        num_train_epochs=float(cfg["train"]["num_train_epochs"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        per_device_train_batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["train"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),
        logging_steps=int(cfg["train"]["logging_steps"]),
        eval_steps=int(cfg["train"]["eval_steps"]),
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        save_steps=int(cfg["train"].get("save_steps", 200)),
        bf16=bool(cfg["model"].get("bf16", True)),
        report_to=("wandb" if use_wandb else "none"),
    )

    trainer.add_callback(
        PairwiseMetricsCallback(
            eval_dataset=eval_ds,
            tokenizer=tok,
            eval_batch_size=int(cfg["train"]["per_device_eval_batch_size"]),
            length_penalty_beta=float(cfg["reward"].get("length_penalty_beta", 0.0)),
            use_wandb=use_wandb,
            preview_rows=int(cfg["run"].get("preview_rows", 8)),
            train_dataset=train_ds,
            train_subset_size=int(cfg["run"].get("train_metrics_subset_size", 2048)),
            train_batch_size=int(cfg["train"]["per_device_eval_batch_size"]),
            train_subset_seed=int(cfg["run"].get("train_metrics_subset_seed", 123)),
        )
    )

    if cfg["run"].get("eval_before_train", True):
        n = int(cfg["run"].get("eval_init_subset", 0))
        if n > 0:
            eval_small = eval_ds.select(range(min(n, len(eval_ds))))
            print(f"[rm] initial eval em subset de {len(eval_small)} exemplos…")
            metrics0 = trainer.evaluate(eval_dataset=eval_small)
        else:
            print("[rm] initial eval (conjunto completo)…")
            metrics0 = trainer.evaluate()
        print("[rm] initial eval metrics:", {k: float(v) for k, v in metrics0.items()})

    print("[rm] training…")
    trainer.train()
    trainer.save_model(cfg["run"]["output_dir"])
    print("[rm] done.")


if __name__ == "__main__":
    main()
