LAAI Humanoid Cycling project

Teaching the Booster K1 humanoid to ride a balance bike: train in MuJoCo on a GPU cluster, then transfer to the real robot.

Robot login (IP, user, password): see `handover docs/credentials.md` (gitignored, ask a teammate), described under "Robot access and credentials" below.

### Commands by platform

Use [docs/commands.md](docs/commands.md) for separate Linux host, macOS host,
Linux GPU cluster, and robot commands. Run commands from the repository root
unless an example explicitly changes directory.

### Setup

Install `uv` on **Linux or macOS**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The project uses **Python 3.12**. Get Booster's code at the pinned versions,
install the simulation and training groups, and run the checks.

#### Linux host

```bash
bash scripts/setup_booster.sh
uv sync --group sim --group train
uv run pre-commit install
uv run pytest
```

#### macOS host

```bash
bash scripts/setup_booster.sh
uv sync --group sim --group train
uv run pre-commit install
uv run pytest
```

On Apple Silicon, use an arm64 Python. See the [macOS setup notes](docs/commands.md#macos-host)
if uv selects an Intel interpreter.

Dependency groups (`pyproject.toml`): `sim` runs Booster's policies in the MuJoCo viewer, `train` is mjlab and W&B (real training on the Linux GPU cluster; on a Mac it runs on CPU for tests and smoke runs), `robot` is Booster's robot SDK (`boosteros[brain]`; Linux only, since Booster publishes no Mac build of its SDK), and `dev` (always installed) has pytest, ruff and pre-commit.

### Training

Tasks are mjlab tasks defined in `src/k1bike/tasks/`. So far: `K1Bike-Walk-Flat` (phase 1, walking on flat ground).

Use the [Linux GPU training and CPU smoke-run commands](docs/commands.md#train-our-mjlab-task).
CPU smoke runs use ordinary Python on both Linux and macOS. Full training runs on
a Linux NVIDIA machine; cluster submission uses SLURM (see [cluster/README.md](cluster/README.md)).

For playback, use the [separate Linux and macOS examples](docs/commands.md#watch-our-trained-policy).
The native MuJoCo viewer uses `python` on Linux and `mjpython` on macOS.

Each run saves checkpoints, an exported ONNX policy and its full config in `logs/rsl_rl/<experiment>/<run>/` (gitignored). See `configs/README.md` for overriding settings and W&B.

### Booster toolchain (gitignored)

`scripts/setup_booster.sh` finds existing Booster checkouts at the project root
first, then checks the older `booster/` subfolder. The project directory itself
can have any name. New clones use this layout:

```
<project root>/
├── pyproject.toml
├── booster_assets/   (K1 robot model)
└── booster_deploy/   (runs policies in MuJoCo or on the robot)
```

Existing checkouts are preserved; setup reports a warning if their commits differ
from the pins. For nested checkouts, it creates top-level relative symlinks so
uv can use its `booster_assets` dependency path without cloning another copy.

To update a version, change the commit in the script, rerun it, and check that the tests pass. Once we fork `booster_deploy` for our own tasks, set `BOOSTER_DEPLOY_REPO` in the script to the fork.

Watch Booster's ready-trained K1 walk using the [platform-specific pretrained-policy
commands](docs/commands.md#run-boosters-pretrained-policy). The optional wrapper
`scripts/run_sample_task.sh` selects the viewer launcher for Linux or macOS.

Note: the handover PDF uses `~/booster/...` paths and a conda env. Resolve commands
from this project's root and use its uv environment; do not assume that its root
directory is named `booster`.

### Project layout

```
assets/bike/          balance-bike model (placeholder dimensions; measure the real bike)
src/k1bike/           our code: paths, scene builder (K1 + bike), robots/k1.py, tasks/ per phase
configs/              how to configure runs and W&B conventions
scripts/              setup_booster.sh, view_model.py, train.py, play.py
cluster/              GPU cluster notes and SLURM job template
tests/                quick checks that the models load and simulate (run in CI)
docs/decisions.md     decision log (what we chose and why)
docs/commands.md      command reference, separated by OS and execution target
AGENTS.md            shared instructions for coding agents
handover docs/        handover notes; credentials.md and PDFs are gitignored
```

View the models with the [Linux or macOS viewer commands](docs/commands.md#view-the-bike-or-the-k1-and-bike).
The `k1_bike` scene places the robot next to the bike; seated interaction is not implemented yet.

### Decisions

Decisions and their reasons are logged in `docs/decisions.md`. So far:

- **Training framework: mjlab** (Isaac Lab-style API on MuJoCo Warp). Training runs on the Linux GPU cluster (`uv sync --group train`); Macs view and test policies in MuJoCo.

Still open:

- **Cluster details.** Fill in `cluster/README.md` once we have access.

### Robot access and credentials

Robot login details (SSH user, password, IP, Wi-Fi) live in `handover docs/credentials.md`. That file is gitignored, so ask a teammate for it. To create your own, copy `handover docs/credentials.example.md` to `handover docs/credentials.md` and fill in the values. Never put passwords in committed files.

The robot firmware must be v1.7.2 or newer before it can run our own policies
(check with `cat /opt/booster/version.txt` after SSH, on the robot's Linux system).
See [robot access commands](docs/commands.md#robot-access) for the distinction
between Linux-only SDK commands and SSH from either host OS.

### Troubleshooting

- See [platform-specific troubleshooting](docs/commands.md#platform-specific-troubleshooting)
  for the Linux GTK/Wayland workaround, Linux headless rendering, and the team's
  macOS hidden-file workaround.

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

- `scripts/robot_info.py`: connects to the real K1 with `boosteros` and prints its model, mode and joint count. Run from a Linux environment on the robot's network; see the command reference.
- `scripts/run_sample_task.sh`: Linux/macOS wrapper for Booster's pretrained MuJoCo tasks; defaults to `k1_fight` and accepts a task name plus deployment options.
- `scripts/connect-booster-k1.sh`, `scripts/mujoco-script.sh`: legacy setup experiments. Use `scripts/setup_booster.sh` and the command reference for the current pinned workflow.
- The ROS installer and Docker rebuild scripts under `scripts/` belong to an optional Ubuntu-container experiment. They are separate from the current mjlab workflow; see [legacy scripts](docs/commands.md#legacy-scripts).
