from src.load_cfg import load_cfg
from src.data import build_dataset
from src.gen import build_generation_kwargs
from src.eval import PeriodicEvalCallback

from src.models.policy import load_policy
from src.models.rm import load_rm, make_reward_fn
from src.models.test import quick_smoke_test

import torch
from trl import GRPOConfig, GRPOTrainer


# ---------------- cfg ----------------
cfg = load_cfg(path="config/train.yaml")

dataset_name = cfg["dataset"]["name"]
split        = cfg["dataset"]["split"]

policy_dir = cfg["policy"]["model_path"]
rm_base    = cfg["rm"]["base"]
rm_adapter = cfg["rm"]["adapter"]
rm_maxlen  = int(cfg["rm"].get("max_length", 2048))

# ---------------- datasets ----------------
print("Carregando datasets...")
ds      = build_dataset(dataset_name=dataset_name, split=split)
ds_test = build_dataset(dataset_name=dataset_name, split="test_prefs")
print("Datasets prontos.\n")

# ---------------- modelos ----------------
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

policy_model, policy_tok = load_policy(policy_dir, dtype=dtype)
rm_model, rm_tok = load_rm(rm_base, rm_adapter, dtype=dtype)

if hasattr(policy_model, "generation_config") and hasattr(policy_model.generation_config, "cache_implementation"):
    policy_model.generation_config.cache_implementation = None

# ---------------- geração ----------------
generation_kwargs = build_generation_kwargs(cfg, policy_tok)

# ---------------- teste ----------------
quick_smoke_test(policy_model, policy_tok, rm_model, rm_tok, generation_kwargs)

# ---------------- reward_fn ----------------
reward_fn = make_reward_fn(rm_model, rm_tok, max_length=rm_maxlen, batch_size=8)

# ---------------- avaliação periódica ----------------
eval_cb = PeriodicEvalCallback(
    eval_every_steps=int(cfg["eval"]["every_steps"]),
    rm_model=rm_model,
    rm_tokenizer=rm_tok,
    test_dataset=ds_test,
    tokenizer=policy_tok,
    n_samples=int(cfg["eval"]["n_samples"]),
    batch_size=int(cfg["eval"]["batch_size"]),
    generation_kwargs=generation_kwargs,
    examples_k=4
)

# ---------------- GRPO ----------------
grpo_cfg = GRPOConfig(
    output_dir=cfg["grpo"]["output_dir"],
    per_device_train_batch_size=int(cfg["grpo"]["per_device_train_batch_size"]),
    gradient_accumulation_steps=int(cfg["grpo"]["gradient_accumulation_steps"]),
    learning_rate=float(cfg["grpo"]["learning_rate"]),
    weight_decay=float(cfg["grpo"]["weight_decay"]),
    num_generations=int(cfg["grpo"]["num_generations"]),
    logging_steps=int(cfg["grpo"]["logging_steps"]),
    save_steps=int(cfg["grpo"]["save_steps"]),
    seed=int(cfg["grpo"]["seed"]),
    report_to=cfg["grpo"]["report_to"],
    generation_kwargs=generation_kwargs,
    save_total_limit = int(cfg["grpo"]["save_total_limit"])
)

trainer = GRPOTrainer(
    model=policy_model,
    args=grpo_cfg,
    train_dataset=ds,
    reward_funcs=[reward_fn],
    processing_class=policy_tok,
)
trainer.add_callback(eval_cb)
eval_cb.trainer = trainer 

print("Trainer criado. Iniciando treino...\n")
trainer.train()
print("\nFIM DO SCRIPT")
