#!/bin/bash
# Linux GPU cluster only; submit from the repository root on the login node.
# Training job template. TODO: set partition, GPU type and time limit for our cluster
# (see cluster/README.md), then submit with:
#   mkdir -p logs
#   sbatch cluster/train.slurm LAAI-K1-Walk-Flat --agent.run-name walk-baseline-ep
# Everything after the task name is passed to mjlab's train command.
#SBATCH --job-name=k1-train
#SBATCH --partition=TODO
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch cluster/train.slurm <task-id> [mjlab options]}"
shift

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export MUJOCO_GL=egl            # headless rendering for rollout videos
# W&B: run `wandb login` once on the login node (see cluster/README.md). Without an API key,
# scripts/train.py logs offline to logs/wandb/. If compute nodes have no internet, uncomment:
# export WANDB_MODE=offline     # then on the login node: uv run --group train wandb sync logs/wandb/offline-run-*
# Runs go to our W&B team, not a personal account. Use the name as it appears in the team's
# URL (wandb.ai/<name>); a WANDB_ENTITY already set in the environment wins.
export WANDB_ENTITY="${WANDB_ENTITY:-K1Booster}"

nvidia-smi
uv run --group train python scripts/train.py "$TASK" --env.scene.num-envs 4096 "$@"
