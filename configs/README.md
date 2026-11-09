# Training configuration

mjlab configures tasks in Python, not YAML. Each task's settings (rewards, randomization
ranges, command ranges, PPO settings) live in its module under `src/k1bike/tasks/`,
for example `src/k1bike/tasks/walk.py`. Change defaults there, in a commit, so runs can be
traced back to code.

For one-off experiments, override any setting on the command line instead of editing code.
All examples below run from the repository root. See [the command reference](../docs/commands.md)
for separate Linux/macOS viewer commands and CPU smoke runs.

## Linux NVIDIA host or allocated GPU node

```bash
uv run --group train python scripts/train.py K1Bike-Walk-Flat \
  --env.scene.num-envs 4096 --agent.run-name walk-lowfriction-ep
```

## Linux and macOS: inspect settings

This command does not open a viewer, so both platforms use ordinary Python:

```bash
uv run --group train python scripts/train.py K1Bike-Walk-Flat --help
```

mjlab saves the full resolved config of every run under `logs/rsl_rl/<experiment>/<run>/params/`.

## Experiment tracking (W&B)

- One shared W&B project (TODO: create it and invite the team); set it with `--agent.wandb-project`.
- Log in once per machine (Linux and macOS): `uv run --group train wandb login`.
- Run names: `<task>-<short-change>-<initials>`, for example `walk-highfriction-ep`.
- mjlab logs the config, the git commit and rollout videos.

## Where things go

| What | Where | In git? |
|---|---|---|
| Task settings | `src/k1bike/tasks/*.py` | yes |
| Checkpoints, exported ONNX, logs | `logs/rsl_rl/<experiment>/<run>/` | no (gitignored) |
| Policies worth keeping | W&B artifacts, or the cluster's project storage | no |
| Policy to deploy | exported ONNX added as a task in the `booster_deploy` fork | in the fork |
