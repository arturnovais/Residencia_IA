#!/usr/bin/env bash
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --job-name=rm
#SBATCH --output=out/jobs/job.%j.out
#SBATCH --error=out/jobs/job.%j.err

set -euo pipefail

APPT_IMAGE="/raid/amadeus/artur_laudite/images/pytorch-2.8.0-cu128-devel.sif"
CONFIG="${CONFIG:-config/rm_config.yaml}"

PROJECT_DIR="$(pwd)"
mkdir -p out/jobs out/home out/wandb out/cache/hf out/cache/pip out/cache/torch out/cache/cuda "out/tmp/${SLURM_JOB_ID:-local}" out/venv

[[ -f "$APPT_IMAGE" ]] || { echo "Imagem não encontrada: $APPT_IMAGE"; exit 1; }
[[ -f "$CONFIG"     ]] || { echo "Config não encontrada: $CONFIG"; exit 1; }
[[ -f "train.py"    ]] || { echo "train.py não encontrado em $(pwd)"; exit 1; }

echo "[run] imagem=$APPT_IMAGE | config=$CONFIG"

srun apptainer exec --nv -B "$PROJECT_DIR":/w/reward_model "$APPT_IMAGE" bash -lc '
  set -euo pipefail
  cd /w/reward_model

  export PYTHONUNBUFFERED=1
  export PYTHONPATH=/w/reward_model:${PYTHONPATH:-}
  export HOME=/w/reward_model/out/home
  export HF_HOME=/w/reward_model/out/cache/hf
  export TRANSFORMERS_CACHE=/w/reward_model/out/cache/hf
  export XDG_CACHE_HOME=/w/reward_model/out/cache/hf
  export PIP_CACHE_DIR=/w/reward_model/out/cache/pip
  export TORCH_HOME=/w/reward_model/out/cache/torch
  export CUDA_CACHE_PATH=/w/reward_model/out/cache/cuda
  export WANDB_DIR=/w/reward_model/out/wandb
  export TMPDIR=/w/reward_model/out/tmp/'"${SLURM_JOB_ID:-local}"'

  if [ -f .env ]; then set -a; . ./.env; set +a; fi

  VENV=out/venv
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV" || (python3 -m ensurepip --upgrade && python3 -m venv "$VENV")
  fi
  . "$VENV/bin/activate"
  python -m pip install -U pip wheel >/dev/null
  python -m pip install -r requirements.txt >/dev/null

  echo "[ctr] gpus visíveis:"
  python - <<PY
import torch; print(torch.cuda.device_count())
PY

  unset RANK WORLD_SIZE LOCAL_RANK
  python /w/reward_model/train.py --config /w/reward_model/'"$CONFIG"'
'
