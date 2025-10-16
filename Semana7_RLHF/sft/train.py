import os
import math
from src.utils import load_cfg, set_seed, enable_tf32
from src.data import build_datasets
from src.model import build_model, build_tokenizer
from src.trainer import build_sft_trainer


def main():
    cfg = load_cfg()
    set_seed(int(cfg["run"]["seed"]))
    enable_tf32()

    rank = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)
    if cfg["run"].get("report_to", "none") == "wandb" and not is_main:
        os.environ["WANDB_MODE"] = "disabled"

    tok = build_tokenizer(
        model_name=cfg["model"]["name"],
        tokenizer_name=cfg["model"].get("tokenizer_name"),
        max_len=int(cfg["data"]["max_seq_length"]),
    )

    model = build_model(
        model_name=cfg["model"]["name"],
        bf16=bool(cfg["model"].get("bf16", True)),
        lora_cfg=cfg["model"].get("lora"),
        tokenizer=tok,
    )

    emb = model.get_input_embeddings().weight
    assert len(tok) == emb.size(0), "Tokenizer and embedding size mismatch"
    assert tok.pad_token_id is not None, "pad_token_id must be set"

    train_ds, eval_ds = build_datasets(cfg, tok)
    trainer = build_sft_trainer(cfg, model, tok, train_ds, eval_ds)

    if cfg["run"].get("report_to", "none") == "wandb" and is_main:
        from transformers.integrations import WandbCallback
        trainer.add_callback(WandbCallback())
        try:
            import wandb
            w = cfg["run"].get("wandb", {})
            wandb.init(
                project=w.get("project", "sft"),
                name=w.get("run_name"),
                config=cfg,
                reinit=True,
            )
        except Exception as e:
            print("[wandb] skip:", e)

    print("[sft] train start")
    trainer.train()
    trainer.save_model(cfg["run"]["output_dir"])
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        try:
            metrics["eval_ppl"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            pass
    print("[sft] done | metrics:", metrics)


if __name__ == "__main__":
    main()
