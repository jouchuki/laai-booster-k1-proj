# GPU cluster

This page's commands run on the **Linux GPU cluster**, from the repository root.
macOS can be the SSH client, but these commands execute on the remote Linux system.
Fill in the table once we have access, then adapt `train.slurm`.
For local Linux/macOS viewing and CPU smoke runs, use [docs/commands.md](../docs/commands.md).

## Cluster details (TODO)

| Question | Answer |
|---|---|
| Login host / how to connect | |
| Scheduler (SLURM?) and partition name | |
| GPU type and how many per job | |
| CUDA driver version (`nvidia-smi`) | |
| Max job time | |
| Containers allowed? (Apptainer/Singularity, Docker) | |
| Project storage path (for checkpoints) | |
| Internet on compute nodes? (needed for W&B and `uv sync`) | |

## First-time setup on the cluster

Install uv on the Linux cluster first if it is not available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, on the Linux cluster:

```bash
git clone https://github.com/jouchuki/laai-booster-k1-proj.git
cd laai-booster-k1-proj
bash scripts/setup_booster.sh assets # only the robot model is needed for training
uv sync --group train
uv run --group train wandb login
uv run --group train pytest          # checks the models load and the walk task builds
```

Submit from the Linux cluster login node after replacing the SLURM placeholders:

```bash
mkdir -p logs
sbatch cluster/train.slurm K1Bike-Walk-Flat --agent.run-name walk-baseline-ep
```

Checkpoints and the exported ONNX policy land in `logs/rsl_rl/k1_walk/<date>_<run-name>/`.

Headless rendering (for rollout videos) needs `MUJOCO_GL=egl`, which `train.slurm` sets.
The headless job uses ordinary Python, not the macOS `mjpython` viewer launcher.
