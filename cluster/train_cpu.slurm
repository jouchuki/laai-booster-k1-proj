#!/bin/bash
# Linux GPU cluster (Hex), CPU-only job; submit from the repository root.
#   mkdir -p logs
#   sbatch --job-name=sit-cpu-smoke --time=00:15:00 cluster/train_cpu.slurm \
#     LAAI-K1-Sit-Puky --env.scene.num-envs 8 --agent.max-iterations 3 --agent.logger wandb
#SBATCH --job-name=k1-train-cpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
TASK="${1:?usage: sbatch cluster/train_cpu.slurm <task-id> [mjlab options]}"
shift

: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run it on the login node.}"
case "${SLURM_JOB_PARTITION:-}" in
    batch|short|long) ;;
    *) echo "Use a student CPU partition: batch, short or long." >&2; exit 1 ;;
esac
if [[ ${SLURM_JOB_NUM_NODES:-0} != 1 || ${SLURM_NTASKS:-0} != 1 ||
      ${SLURM_GPUS_ON_NODE:-0} != 0 || -n ${SLURM_JOB_GPUS:-} ||
      ! ${SLURM_CPUS_PER_TASK:-0} =~ ^[1-9][0-9]*$ ]]; then
    echo "This CPU job requires one node, one task, positive CPU count and no allocated GPUs." >&2
    exit 1
fi
for argument in "$@"; do
    case "$argument" in
        --gpu-ids|--gpu-ids=*)
            echo "This launcher sets --gpu-ids None; omit GPU options." >&2
            exit 1 ;;
    esac
done

cd "$SLURM_SUBMIT_DIR"
if [[ ! -f pyproject.toml || ! -f scripts/train.py || ! -x .venv/bin/python ]]; then
    echo "Submit from the repository root after preparing .venv; see cluster/README.md." >&2
    exit 1
fi
export PATH="$PATH:$HOME/.local/bin"
export UV_PYTHON_DOWNLOADS=never
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_ENTITY="${WANDB_ENTITY:-K1Booster}"
mkdir -p logs
printf 'CPU job %s on %s: %s CPUs, no GPU\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$SLURM_CPUS_PER_TASK"

uv_run=(run --no-sync --group train)
if [[ -f .env ]]; then
    uv_run+=(--env-file .env)
fi
# Same prepared training dependencies as GPU jobs; --gpu-ids None forces CPU execution.
# W&B logs offline without credentials; --agent.logger tensorboard skips W&B entirely.
exec uv "${uv_run[@]}" python scripts/train.py "$TASK" \
    --gpu-ids None --env.scene.num-envs 256 "$@"
