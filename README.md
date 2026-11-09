LAAI Humanoid Cycling project

Teaching the Booster K1 humanoid to ride a balance bike: train in MuJoCo on a GPU cluster, then transfer to the real robot.

Robot login (IP, user, password): see `handover docs/credentials.md` (gitignored, ask a teammate), described under "Robot access and credentials" below.

### Setup (Mac)

1. Install the `uv` package manager:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. From the repo folder, get Booster's code at the versions the team has pinned, then install everything:

```
scripts/setup_booster.sh
uv sync --group sim --group train
uv run pre-commit install
uv run pytest
```

If `uv` picks an Intel (x86_64) Python on an Apple Silicon Mac, create the environment with an arm64 Python first, for example `uv venv --python /opt/homebrew/bin/python3.12`, then rerun the `uv sync` line.

Dependency groups (`pyproject.toml`): `sim` runs Booster's policies in the MuJoCo viewer, `train` is mjlab and W&B (real training on the Linux GPU cluster; on a Mac it runs on CPU for tests and smoke runs), `robot` is Booster's robot SDK (`boosteros[brain]`; Linux only, since Booster publishes no Mac build of its SDK), and `dev` (always installed) has pytest, ruff and pre-commit.

### Training

Tasks are mjlab tasks defined in `src/k1bike/tasks/`. So far: `K1Bike-Walk-Flat` (phase 1, walking on flat ground).

```
# On the cluster (see cluster/README.md):
sbatch cluster/train.slurm K1Bike-Walk-Flat --agent.run-name walk-baseline-ep

# Quick smoke test on a Mac (CPU, a few robots, a few iterations):
uv run python scripts/train.py K1Bike-Walk-Flat --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 --agent.logger tensorboard

# Watch a trained policy:
uv run python scripts/play.py K1Bike-Walk-Flat --checkpoint-file logs/rsl_rl/k1_walk/<run>/model_<n>.pt
```

Each run saves checkpoints, an exported ONNX policy and its full config in `logs/rsl_rl/<experiment>/<run>/` (gitignored). See `configs/README.md` for overriding settings and W&B.

### Booster toolchain (`booster/`, gitignored)

`scripts/setup_booster.sh` clones Booster's repos into `booster/` at pinned commits, so everyone uses the same robot model and deploy code:

```
booster/
├── booster_assets/   (K1 robot model)
└── booster_deploy/   (runs policies in MuJoCo or on the robot)
```

To update a version, change the commit in the script, rerun it, and check that the tests pass. Once we fork `booster_deploy` for our own tasks, set `BOOSTER_DEPLOY_REPO` in the script to the fork.

Watch Booster's ready-trained K1 walk in the simulator (on a Mac the viewer needs `mjpython`):

```
cd booster/booster_deploy
uv run mjpython scripts/deploy.py --task k1_walk --mujoco
```

Note: the handover PDF uses `~/booster/...` paths and a conda env; in this repo they are `booster/...` and the uv environment instead.

### Project layout

```
assets/bike/          balance-bike model (placeholder dimensions; measure the real bike)
src/k1bike/           our code: paths, scene builder (K1 + bike), robots/k1.py, tasks/ per phase
configs/              how to configure runs and W&B conventions
scripts/              setup_booster.sh, view_model.py, train.py, play.py
cluster/              GPU cluster notes and SLURM job template
tests/                quick checks that the models load and simulate (run in CI)
docs/decisions.md     decision log (what we chose and why)
handover docs/        handover notes; credentials.md and PDFs are gitignored
```

View the models: `uv run mjpython scripts/view_model.py bike --push 3` or `uv run mjpython scripts/view_model.py k1_bike`.

### Decisions

Decisions and their reasons are logged in `docs/decisions.md`. So far:

- **Training framework: mjlab** (Isaac Lab-style API on MuJoCo Warp). Training runs on the Linux GPU cluster (`uv sync --group train`); Macs view and test policies in MuJoCo.

Still open:

- **Cluster details.** Fill in `cluster/README.md` once we have access.

### Robot access and credentials

Robot login details (SSH user, password, IP, Wi-Fi) live in `handover docs/credentials.md`. That file is gitignored, so ask a teammate for it. To create your own, copy `handover docs/credentials.example.md` to `handover docs/credentials.md` and fill in the values. Never put passwords in committed files.

The robot firmware must be v1.7.2 or newer before it can run our own policies (check with `cat /opt/booster/version.txt` on the robot).

### Troubleshooting

- `ModuleNotFoundError: booster_assets` (or `k1bike`) on macOS: macOS sometimes marks files in `.venv` as hidden, and Python then skips them. Fix with `chflags -R nohidden .venv`.

### Initial plan

1. Prove sim-to-real with walking. Train a standard walking policy for our robot and run it on the real one. This tests the whole pipeline (model, simulator, training, deployment), and walking is also the base skill for riding a balance bike.
2. Build the balance-bike model in simulation: frame, handlebar and fork, two wheels, saddle. No pedals.
3. Sit on a supported bike. The bike is held upright; the robot learns to sit stably with hands on the bars and feet on the ground.
4. Walk the bike, supported. Still held upright, the robot pushes off with alternating feet to roll forward.
5. Free bike, feet as support. Remove the support. The robot strides forward and can put a foot down to catch itself, while learning to balance by steering.
6. Simulation to real robot. Carefully test on real robot

extra things: Stride and glide. Lift both feet and coast, balancing by steering only, then push again, steering on command. Ride in a requested direction.

### Resources

- K1 software overview: https://www.roboticscenter.ai/hardware/booster-k1/software
- Booster's repos: [booster_assets](https://github.com/BoosterRobotics/booster_assets), [booster_deploy](https://github.com/BoosterRobotics/booster_deploy), [booster_train](https://github.com/BoosterRobotics/booster_train)
- [mjlab](https://github.com/mujocolab/mjlab) (training framework)

### Team scripts (from PR #5)

- `scripts/robot_info.py`: connects to the real K1 with `boosteros` and prints its model, mode and joint count. Run it on the robot's network: `uv run --group robot python scripts/robot_info.py`.
- `connect-booster-k1.sh`, `mujoco-script.sh`, `run_sample_task.sh`: earlier setup and run scripts. `scripts/setup_booster.sh` and the commands above cover the same steps with pinned versions.
