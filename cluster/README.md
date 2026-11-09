# GPU cluster

Training runs here, not on a Mac. Fill in the table once we have access, then adapt
`train.slurm`.

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

```
git clone https://github.com/jouchuki/laai-booster-k1-proj.git
cd laai-booster-k1-proj
scripts/setup_booster.sh assets     # only the robot model is needed for training
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group train
uv run wandb login
uv run pytest                        # checks the models load and the walk task builds
```

Then submit a job: `sbatch cluster/train.slurm K1Bike-Walk-Flat --agent.run-name walk-baseline-ep`.

Checkpoints and the exported ONNX policy land in `logs/rsl_rl/k1_walk/<date>_<run-name>/`.

Headless rendering (for rollout videos) needs `MUJOCO_GL=egl`, which `train.slurm` sets.
