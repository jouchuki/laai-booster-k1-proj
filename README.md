# LAAI Humanoid Cycling project

Teaching the Booster K1 humanoid to ride a balance bike: train in simulation, then run on the real robot.

## Plan and status

| Phase | Goal | Status | Where |
|---|---|---|---|
| 1 | Walking: prove sim-to-real | Walks and turns on the robot (run 2, 2026-11-11); ankle-roll oscillation at high speed | [experiments/walk/](experiments/walk/README.md) |
| 2 | Balance-bike model in simulation | Measured USDZ-derived model, provisional physics | [models/bike/](models/bike/README.md), [docs/bike-dimensions.md](docs/bike-dimensions.md) |
| 3 | Sit on a supported bike | Seated scene ready (hands welded to the grips, pelvis on the saddle by contact); no training task yet | [models/k1_on_bike/](models/k1_on_bike/README.md) |
| 4 | Walk the bike, supported | Not started | |
| 5 | Free bike, feet as support | Not started | |
| 6 | Simulation to real robot | Robot pipeline works for phase 1 | [robot/](robot/README.md) |

1. Prove sim-to-real with walking. Train a standard walking policy for our robot and run it on the real one. This tests the whole pipeline (model, simulator, training, deployment), and walking is also the base skill for riding a balance bike.
2. Build the balance-bike model in simulation: frame, handlebar and fork, two wheels, saddle. No pedals.
3. Sit on a supported bike. The bike is held upright; the robot learns to sit stably with hands on the bars and feet on the ground.
4. Walk the bike, supported. Still held upright, the robot pushes off with alternating feet to roll forward.
5. Free bike, feet as support. Remove the support. The robot strides forward and can put a foot down to catch itself, while learning to balance by steering.
6. Simulation to real robot. Carefully test on the real robot.

Extra: stride and glide. Lift both feet and coast, balancing by steering only, then push again, steering on command. Ride in a requested direction.

## Where everything is

```
models/              everything the simulator loads
  k1/                the Booster K1 robot (Booster's model, pinned; settings in src/k1bike/robots/k1.py)
  bike/              the balance bike we use (USDZ-derived), with its source USDZ
  bike_primitive/    the old placeholder bike, kept for comparison
  k1_on_bike/        the K1 seated on the bike: solved pose
experiments/         one folder per skill (walk, later sit, push, ride)
  walk/              training task, export, booster_deploy task files
    runs/runN/       record of each training run: notes, task snapshot, policy, robot logs
logs/                raw training output, gitignored: logs/laai_<experiment>/<date>_<run-name>/
src/k1bike/          shared Python code: model paths, K1 settings, scenes, seating
scripts/             tools: setup, viewer, train/play, model generators
robot/               tools for the real robot: install into booster_deploy, copy, dry run, info
cluster/             SLURM job template for the GPU cluster
docs/                commands, decisions, bike measurements, handover notes
tests/               quick checks that the models load and the tasks build (run in CI)
examples/            practice scripts (Stable-Baselines3), not part of the pipeline
legacy/              old, unvalidated experiments (ROS 2 Docker)
```

| Folder | Contents |
|---|---|
| [models/](models/README.md) | `k1/README.md` (where the robot model comes from), `bike/` (`bike.xml`, `meshes/`, `measurements.json`, `source/`), `bike_primitive/bike.xml`, `k1_on_bike/seated_pose.json` |
| [experiments/](experiments/README.md) | `<experiment>/task.py` (mjlab task `LAAI-K1-<Experiment>-...`), `export.py`, `deploy/` (booster_deploy tasks `laai_<experiment>...`), `runs/runN/`, `README.md` (experiment report) |
| `logs/` | Checkpoints, TensorBoard and videos from `scripts/train.py`; link the ones you keep from the run's `notes.md` |
| `src/k1bike/` | `paths.py` (every model and log path), `robots/k1.py` (K1 model and booster_deploy settings), `scene.py` (bike, K1 + bike and seated scenes), `seat.py` (hand welds, seat contact) |
| `scripts/` | `setup_booster.sh`, `booster_paths.sh` (Booster checkouts), `view_model.py` (viewer), `train.py`, `play.py` (any experiment), `run_sample_task.sh` (Booster's pretrained policies), `convert_bike_usdz.py` (makes `models/bike/`), `solve_seated_pose.py` (makes `models/k1_on_bike/`) |
| [robot/](robot/README.md) | `install.sh <experiment>`, `setup_robot.sh`, `dry_run.py` (policy on live sensors, no motor commands), `robot_info.py` |
| [cluster/](cluster/README.md) | `train.slurm` (cluster details still to fill in) |
| `docs/` | [commands.md](docs/commands.md) (every command by platform), [decisions.md](docs/decisions.md), [bike-dimensions.md](docs/bike-dimensions.md), [handover/](docs/handover/README.md) (first setup, credentials template) |
| `booster_assets/`, `booster_deploy/` | Booster's code, cloned by `scripts/setup_booster.sh` (gitignored) |
| `AGENTS.md`, `CLAUDE.md` | Shared instructions for coding agents |

Names use the project prefix and the experiment, never "bike" for skills that are not on the
bike: mjlab task `LAAI-K1-<Experiment>-...`, log folder and booster_deploy task `laai_<experiment>`.

The robot never passes through itself or the bike: every experiment keeps the K1's
self-collision, and the robot collides with every part of the bike except where its welded
forearms meet the fork (see [decisions](docs/decisions.md)). Tests check both.

## Commands by platform

Use [docs/commands.md](docs/commands.md) for separate Linux host, macOS host,
Linux GPU cluster, and robot commands. Run commands from the repository root
unless an example explicitly changes directory.

## Setup

Install `uv` on **Linux or macOS**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The project uses **Python 3.12**. Get Booster's code at the pinned versions,
install the simulation and training groups, and run the checks.

### Linux host

```bash
bash scripts/setup_booster.sh
uv sync --group sim --group train
uv run pytest
```

### macOS host

```bash
bash scripts/setup_booster.sh
uv sync --group sim --group train
uv run pytest
```

On Apple Silicon, use an arm64 Python. See the [macOS setup notes](docs/commands.md#macos-host)
if uv selects an Intel interpreter. If the project folder syncs with iCloud, see the
[hidden-file fix](docs/commands.md#macos-hidden-environment-files).

### Booster toolchain (gitignored)

`scripts/setup_booster.sh` finds existing Booster checkouts at the project root
first, then checks the older `booster/` subfolder. The project directory itself
can have any name. New clones go to `booster_assets/` and `booster_deploy/` at the
project root, next to `pyproject.toml`.

Existing checkouts are preserved; setup reports a warning if their commits differ
from the pins. For nested checkouts, it creates top-level relative symlinks so
uv can use its `booster_assets` dependency path without cloning another copy.

To update a version, change the commit in the script, rerun it, and check that the tests pass. Once we fork `booster_deploy` for our own tasks, set `BOOSTER_DEPLOY_REPO` in the script to the fork.

Watch Booster's ready-trained K1 walk using the [platform-specific pretrained-policy
commands](docs/commands.md#run-boosters-pretrained-policy). The optional wrapper
`scripts/run_sample_task.sh` selects the viewer launcher for Linux or macOS.

The [handover notes](docs/handover/README.md) use `~/booster/...` paths and a conda env.
Resolve commands from this project's root and use its uv environment; do not assume that its
root directory is named `booster`.

## Robot access

Ask a teammate for the robot login (user, IP, password), and keep it in the gitignored
`docs/handover/credentials.md` (template: `credentials.example.md`). Never commit it.

The robot firmware must be v1.7.2 or newer before it can run our own policies
(check with `cat /opt/booster/version.txt` after SSH, on the robot's Linux system).
See [robot access commands](docs/commands.md#robot-access) for the distinction
between Linux-only SDK commands and SSH from either host OS.

## Troubleshooting

See [platform-specific troubleshooting](docs/commands.md#platform-specific-troubleshooting)
for the Linux GTK/Wayland workaround, Linux headless rendering, and the macOS hidden-file fix.
