#!/usr/bin/env bash
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --nodelist=dgx-H100-03
#SBATCH --job-name=grpo-train
#SBATCH --output=out/jobs/%x.%j.out
#SBATCH --error=out/jobs/%x.%j.err

set -euo pipefail

# logs à prova de -u
log(){ echo "[HOST $(date +'%F %T')] ${*:-}"; }
die(){ echo "[HOST $(date +'%F %T')] ERRO: ${*:-}" >&2; exit 1; }


PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
OUT_DIR="$PROJECT_DIR/out"
mkdir -p "$OUT_DIR" "$OUT_DIR/jobs"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY="$WANDB_API_KEY"
  export WANDB_MODE="online"
  echo "[INFO] WandB em modo online (chave encontrada no .env)"
else
  export WANDB_MODE="offline"
  echo "[INFO] WandB em modo offline (nenhuma chave encontrada)"
fi



IMG="/raid/aluno_artur/images/pytorch-2.8.0-cu128-devel.sqsh"
RUN_NAME="pytorch_ppo"
[[ -f "$IMG" ]] || die "Imagem não encontrada: $IMG"


export ENROOT_CACHE_PATH="/raid/aluno_artur/.enroot/cache"
export ENROOT_DATA_PATH="/raid/aluno_artur/.enroot/data"
export ENROOT_RUNTIME_PATH="/raid/aluno_artur/.enroot/run"
export ENROOT_TEMP_PATH="/raid/aluno_artur/.enroot/tmp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH"
chmod 700 "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH"


HF_HOST="/raid/aluno_artur/hf_cache"
mkdir -p "$HF_HOST"/{datasets,models,metrics}
HF_CONT="/data/out/hf_cache"

# Vars de cache dentro do container
export HF_HOME="$HF_CONT"
export HF_DATASETS_CACHE="$HF_CONT/datasets"
export HF_METRICS_CACHE="$HF_CONT/metrics"
export XDG_CACHE_HOME="$HF_CONT"


export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU bind
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

log "Usando imagem: $IMG"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


if ! enroot list | grep -q "^${RUN_NAME}\$"; then
  log "Criando container persistente '$RUN_NAME' (apenas uma vez)"
  enroot create -n "$RUN_NAME" "$IMG"
  log "Container '$RUN_NAME' criado."
else
  log "Container '$RUN_NAME' já existe. Reutilizando."
fi


srun --ntasks=1 enroot start --rw \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_METRICS_CACHE="$HF_METRICS_CACHE" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env WANDB_PROJECT="${WANDB_PROJECT:-}" \
  --env WANDB_MODE="${WANDB_MODE:-offline}" \
  --mount "$PROJECT_DIR:/data" \
  --mount "$OUT_DIR:/data/out" \
  --mount "$HF_HOST:$HF_CONT" \
  "$RUN_NAME" /bin/bash -lc '
    set -euo pipefail
    cd /data

    echo "[ctr] Python no container:"
    python3 -V || true

    echo "[ctr] Caches HF:"
    df -h /data/hf_cache | tail -n1 || true

    # ====== Instalação rápida (idempotente) ======
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_INPUT=1
    export PIP_NO_COMPILE=1
    export PIP_CACHE_DIR=/data/out/pip_cache
    mkdir -p "$PIP_CACHE_DIR" /data/out/pip_lib

    REQ_FILE=/data/requirements.txt
    REQ_HASH_FILE=/data/out/pip_lib/.requirements.sha256
    if [ -f "$REQ_FILE" ]; then
      NEW_HASH=$(sha256sum "$REQ_FILE" | awk "{print \$1}")
    else
      echo "[ctr] requirements.txt não encontrado, pulando instalação de pacotes."
      NEW_HASH=""
    fi

    OLD_HASH=""
    if [ -f "$REQ_HASH_FILE" ]; then
      OLD_HASH=$(cat "$REQ_HASH_FILE" || true)
    fi

    if [ -f "$REQ_FILE" ]; then
      if grep -Eqi "^(torch|torchvision|torchaudio)([=><!~].*)?$" "$REQ_FILE"; then
        echo "[ctr][ERRO] requirements.txt inclui torch/vision/audio. Remova-os (a imagem já traz torch 2.8.0+cu128)." >&2
        exit 2
      fi
    fi

    if [ "${NEW_HASH:-}" != "${OLD_HASH:-}" ]; then
      echo "[ctr] Instalando/atualizando pacotes locais (mudança detectada em requirements.txt)..."
      python3 -m pip install --break-system-packages --target /data/out/pip_lib -U pip wheel
      python3 -m pip install --break-system-packages --target /data/out/pip_lib -r /data/requirements.txt
      rm -rf /data/out/pip_lib/torch* /data/out/pip_lib/torchvision* /data/out/pip_lib/torchaudio* || true
      echo "$NEW_HASH" > "$REQ_HASH_FILE"
      echo "[ctr] Pacotes prontos (cache: $PIP_CACHE_DIR)."
    else
      echo "[ctr] requirements.txt sem mudanças. Reutilizando /data/out/pip_lib."
    fi
    # ============================================

    echo "[ctr] Copiando pip_lib para SSD local..."
    mkdir -p /tmp/pip_lib

    # fallback automático se rsync não estiver disponível
    if command -v rsync >/dev/null 2>&1; then
      rsync -a /data/out/pip_lib/ /tmp/pip_lib/ || true
    else
      cp -r /data/out/pip_lib/* /tmp/pip_lib/ 2>/dev/null || true
    fi

    export PYTHONPATH=${PYTHONPATH:-}/tmp/pip_lib
    echo "[ctr] PYTHONPATH=$PYTHONPATH"

    python3 - <<PY
import torch
torch.cuda.init()
print("[ctr] CUDA inicializada e pronta.", flush=True)
PY

    python3 - <<PY
import os
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
print("[ctr][hf] token presente:", bool(tok))
PY

    echo "[ctr] Rodando train.py..."
    python3 -u /data/train.py
  '

log "Job finalizado com sucesso."
