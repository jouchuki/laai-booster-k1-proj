#!/bin/bash
# Linux GPU cluster only; submit from the repository root on the login node.
# Hex training job for walk or sit (see cluster/README.md). Submit with:
#   mkdir -p logs
#   sbatch --job-name=sit-stage0 cluster/train.slurm LAAI-K1-Sit-Puky --agent.run-name stage0
# Everything after the task name is passed to mjlab's train command.
#SBATCH --job-name=k1-train
#SBATCH --partition=batch-gpu
#SBATCH --gres=gpu:rtx4500:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch cluster/train.slurm <task-id> [mjlab options]}"
shift

: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run it on the login node.}"
if [[ ${SLURM_JOB_PARTITION:-} != batch-gpu || ${SLURM_JOB_NUM_NODES:-0} != 1 ||
      ${SLURM_NTASKS:-0} != 1 || ${SLURM_GPUS_ON_NODE:-0} != 1 ||
      ! ${SLURM_CPUS_PER_TASK:-0} =~ ^[1-5]$ ]]; then
    echo "This Hex job requires batch-gpu, one node, one task, one GPU and 1-5 CPUs." >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f pyproject.toml || ! -f scripts/train.py ]]; then
    echo "Submit from this project's repository root." >&2
    exit 1
fi
# uv's default installer location may be missing from a non-interactive job's PATH.
export PATH="$PATH:$HOME/.local/bin"
if [[ ! -x .venv/bin/python ]]; then
    echo "Prepare .venv on the login node using cluster/README.md before submitting." >&2
    exit 1
fi
export UV_PYTHON_DOWNLOADS=never
mkdir -p logs
export MUJOCO_GL=egl            # headless rendering for rollout videos
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_ENTITY="${WANDB_ENTITY:-K1Booster}"
printf 'Job %s on %s: %s CPUs, %s GPU\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$SLURM_CPUS_PER_TASK" "$SLURM_GPUS_ON_NODE"
# W&B: use WANDB_API_KEY in the checkout's ignored .env, or log in on the login node.
# uv parses .env as data; never source it as a shell script or print its contents.
uv_run=(run --no-sync --group train)
if [[ -f .env ]]; then
    uv_run+=(--env-file .env)
fi
# Without an API key,
# scripts/train.py logs offline to logs/wandb/. If compute nodes have no internet, uncomment:
# export WANDB_MODE=offline     # sync later on the login node; see cluster/README.md

nvidia-smi
# Install the locked training environment on the login node first. Avoid downloads or
# environment changes on compute nodes; CUDA_VISIBLE_DEVICES is supplied by SLURM.
uv "${uv_run[@]}" python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check the GPU driver and training environment"'
exec uv "${uv_run[@]}" python scripts/train.py "$TASK" --env.scene.num-envs 4096 "$@"
