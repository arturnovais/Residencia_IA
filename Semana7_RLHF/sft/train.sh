#!/usr/bin/env bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --job-name=sft
#SBATCH --output=out/jobs/job.%j.out
#SBATCH --error=out/jobs/job.%j.err

set -euo pipefail

APPT_IMAGE="${APPT_IMAGE:-/raid/amadeus/artur_laudite/images/pytorch-2.8.0-cu128-devel.sif}"
CONFIG="${CONFIG:-config/train_config.yaml}"

PROJECT_DIR="$(pwd)"
JOB_TMP="out/tmp/${SLURM_JOB_ID:-local}"

mkdir -p \
  out/jobs out/home out/wandb \
  out/cache/hf out/cache/pip out/cache/torch out/cache/cuda \
  "$JOB_TMP" out/venv

[[ -f "$APPT_IMAGE" ]] || { echo "Imagem não encontrada: $APPT_IMAGE"; exit 1; }
[[ -f "$CONFIG" ]]     || { echo "Config não encontrada: $CONFIG"; exit 1; }
[[ -f "train.py" ]]    || { echo "train.py não encontrado em $PROJECT_DIR"; exit 1; }

echo "[run] GPUs=${CUDA_VISIBLE_DEVICES:-<SLURM>}"
echo "[run] IMAGE=$APPT_IMAGE"
echo "[run] PWD=$PROJECT_DIR"
echo "[run] CONFIG=$CONFIG"

srun apptainer exec --nv -B "$PROJECT_DIR":/w "$APPT_IMAGE" bash -lc '
  set -euo pipefail
  cd /w

    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/w/src:${PYTHONPATH:-}
    export HOME=/w/out/home
    export HF_HOME=/w/out/cache/hf
    export TRANSFORMERS_CACHE=/w/out/cache/hf
    export XDG_CACHE_HOME=/w/out/cache/hf
    export PIP_CACHE_DIR=/w/out/cache/pip
    export TORCH_HOME=/w/out/cache/torch
    export CUDA_CACHE_PATH=/w/out/cache/cuda
    export WANDB_DIR=/w/out/wandb
    export TMPDIR=/w/'"$JOB_TMP"'

    export TORCHDYNAMO_DISABLE=1          
    export TORCHINDUCTOR_DISABLE=1
    export TORCH_COMPILE_DISABLE=1
    export TRITON_DISABLE_TUNING=1
    export CUDA_LAUNCH_BLOCKING=1
    export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # ================================================================

  if [ -f /w/.env ]; then set -a; . /w/.env; set +a; fi

  VENV=/w/out/venv
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV" || (python3 -m ensurepip --upgrade && python3 -m venv "$VENV")
  fi
  . "$VENV/bin/activate"
  python -m pip install -U pip wheel
  python -m pip install -r /w/requirements.txt

  python - <<PY
import os, torch
print("[container] torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpus:", torch.cuda.device_count())
print("[gpu] CUDA_VISIBLE_DEVICES=", os.getenv("CUDA_VISIBLE_DEVICES"))
for k in ["HOME","HF_HOME","TRANSFORMERS_CACHE","PIP_CACHE_DIR","WANDB_DIR","TMPDIR"]:
    print(f"[paths] {k}=", os.getenv(k))
PY

  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT \
        ACCELERATE_TPU_ACCELERATE ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED

  export OMP_NUM_THREADS=1
  export NCCL_DEBUG=WARN

  python /w/train.py --config /w/'"$CONFIG"'
'
