# LAAI Humanoid Cycling project

Teaching the Booster K1 humanoid to ride a balance bike: train in simulation, then run on the real robot.

## Quick start: watch the real bike roll and steer
### Plan and status

| Phase | Goal | Status | Where |
|---|---|---|---|
| 1 | Walking: prove sim-to-real | Walks and turns on the robot (run 2, 2026-11-11); ankle-roll oscillation at high speed | [phases/walk/](phases/walk/README.md) |
| 2 | Balance-bike model in simulation | Measured Sketchfab model, provisional physics | [assets/bike/](assets/bike/README.md), [docs/bike-dimensions.md](docs/bike-dimensions.md) |
| 3 | Sit on a supported bike | Not started | |
| 4 | Walk the bike, supported | Not started | |
| 5 | Free bike, feet as support | Not started | |
| 6 | Simulation to real robot | Robot pipeline works for phase 1 | [deploy/](deploy/README.md) |

### Initial plan

1. Prove sim-to-real with walking. Train a standard walking policy for our robot and run it on the real one. This tests the whole pipeline (model, simulator, training, deployment), and walking is also the base skill for riding a balance bike.
2. Build the balance-bike model in simulation: frame, handlebar and fork, two wheels, saddle. No pedals.
3. Sit on a supported bike. The bike is held upright; the robot learns to sit stably with hands on the bars and feet on the ground.
4. Walk the bike, supported. Still held upright, the robot pushes off with alternating feet to roll forward.
5. Free bike, feet as support. Remove the support. The robot strides forward and can put a foot down to catch itself, while learning to balance by steering.
6. Simulation to real robot. Carefully test on real robot

extra things: Stride and glide. Lift both feet and coast, balancing by steering only, then push again, steering on command. Ride in a requested direction.

### Where everything is

**Each phase has one folder**, `phases/<phase>/`, with everything for that skill
(see [phases/README.md](phases/README.md)):

| In `phases/<phase>/` | What | Phase 1: [phases/walk/](phases/walk/README.md) |
|---|---|---|
| `README.md` | Phase report: status, timeline, setup, runs, robot-side features | |
| `task.py` | mjlab training task | `LAAI-K1-Walk-Flat` |
| `export.py` | Export a checkpoint to the robot's policy format | `policy.pt` |
| `deploy/` | booster_deploy task files | `laai_walk`, `laai_walk_ankle_kd15` |
| `runs/runN/` | One folder per training run: notes, task snapshot, policy, robot logs | `run1/`, `run2/` |

**Shared tools** are used by every phase:

| Path | What |
|---|---|
| `scripts/train.py`, `scripts/play.py` | Train any phase's task (GPU cluster) and watch a checkpoint |
| `deploy/` | `install.sh <phase>` (into booster_deploy), `setup_robot.sh` (copy to the robot), `dry_run.py` (policy on live sensors, no motor commands) |
| `docs/commands.md` | Every command, separated by Linux host, macOS host, cluster and robot |
| `src/k1bike/` | Shared Python code: model paths, bike and K1 + bike scenes |
| `assets/bike/` | Bike model: original USDZ, scaled MuJoCo derivative, primitive baseline and notes |
| `scripts/` (other) | `setup_booster.sh` and `booster_paths.sh` (Booster checkouts), `view_model.py` (viewer), `run_sample_task.sh` (Booster's pretrained policies), `convert_bike_usdz.py`, `robot_info.py`, legacy scripts |
| `tests/` | Quick checks that the models load and the tasks build (run in CI) |
| `docs/decisions.md` | What we chose and why |
| `docs/bike-dimensions.md` | Physical bike measurements and estimates |
| `docs/setup/handover.md` | First Mac and robot setup (Nov 2026) |
| `examples/rl_basics/` | Practice scripts (Stable-Baselines3), not part of the pipeline |
| `booster_assets/`, `booster_deploy/` | Booster's code, cloned by `scripts/setup_booster.sh` (gitignored) |
| `AGENTS.md`, `CLAUDE.md` | Shared instructions for coding agents |

Names use the project prefix and the phase, never "bike" for skills that are not on the bike:
mjlab task `LAAI-K1-<Phase>-...`, log folder and booster_deploy task `laai_<phase>`.

Run this on a **Linux or macOS desktop**. No GPU or robot is required. Open a
terminal in the cloned repository's root: the folder containing `pyproject.toml`.
All commands below run from that directory. The project uses **Python 3.12**;
uv selects the project interpreter and manages the environment.

### 1. Set up the local viewer

**Linux and macOS**: install `uv` if you do not already have it, then restart
your terminal so `uv` is on `PATH`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Linux and macOS**, from the repository root:

```bash
bash scripts/setup_booster.sh assets
uv sync
```

This fetches the pinned robot assets required by the project and installs the
local viewer dependencies. The generated bike meshes and textures are included
in `assets/bike/real_bike/`; you can view them without running the converter.

### 2. Open the rolling and steering demo

**Linux host**, from the repository root:

```bash
uv run python scripts/view_model.py bike --model real_bike --push 1.5 --steer 12 --paused
```

**macOS host**, from the repository root:

```bash
uv run mjpython scripts/view_model.py bike --model real_bike --push 1.5 --steer 12 --paused
```

The window opens **paused**, with the blue textured bike's front assembly turned
12°. Click the viewer window to give it keyboard focus, then press **Space**.
The bike starts rolling at **1.5 m/s**, with both wheels spinning. Watch the
front wheel and handlebars steer together while the camera follows the bike.
It can turn, slow down and topple as gravity and ground contact act on it.

| Control | Action |
|---|---|
| **Space** | Pause or resume the simulation. |
| **R** | Reset the pose, steering and initial push to replay the demo. If paused, press Space to release it again. |
| Close the window | Stop the viewer before launching another example. |

`--steer` sets the starting steering angle; the steering joint moves freely
after release. `--push` gives an initial rolling speed. This is a passive physics
demo; powered riding, automatic balance and a seated K1 controller are still to
be implemented. The bike's configured mass is **1.5 kg**, a provisional estimate.

### 3. Try straight rolling or a fall

Close the current viewer, then run **one command at a time**. The first gives a
straight initial push. The second releases the bike from a 5° lean, raised 3 cm
to avoid starting inside the floor. Press **Space** to release it and **R** to
repeat.

**Linux host**, from the repository root:

```bash
# Straight initial push, with no initial steering.
uv run python scripts/view_model.py bike --model real_bike --push 1.5 --paused

# Lean-and-fall test.
uv run python scripts/view_model.py bike --model real_bike --lean 5 --drop 0.03 --paused
```

**macOS host**, from the repository root:

```bash
# Straight initial push, with no initial steering.
uv run mjpython scripts/view_model.py bike --model real_bike --push 1.5 --paused

# Lean-and-fall test.
uv run mjpython scripts/view_model.py bike --model real_bike --lean 5 --drop 0.03 --paused
```

Change `--steer 12` to `--steer -12` to try the opposite starting direction,
or change `--push` to compare speeds. Omitting `--model real_bike` loads the
older green bike; `--model primitive` selects the original simple blue shapes.
See the [model notes](assets/bike/real_bike/README.md) for scan and physics
limitations, and [platform troubleshooting](docs/commands.md#platform-specific-troubleshooting)
if the viewer does not open. On macOS, the native viewer requires `mjpython`.

## Development and training setup

For Booster's pretrained policies and our training pipeline, fetch both Booster
checkouts and install the additional dependency groups. From the repository root:

#### Linux host

```bash
bash scripts/setup_booster.sh
uv sync --group sim --group train
uv run pytest
```

#### macOS host

```bash
bash scripts/setup_booster.sh
uv sync --group sim --group train
uv run pytest
```

On Apple Silicon, use an arm64 Python. See the [macOS setup notes](docs/commands.md#macos-host)
if uv selects an Intel interpreter.

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

Note: the handover notes ([docs/setup/handover.md](docs/setup/handover.md)) use `~/booster/...`
paths and a conda env. Resolve commands from this project's root and use its uv environment;
do not assume that its root directory is named `booster`.

```
assets/bike/          primitive, Sketchfab and real_bike MuJoCo bikes, plus model notes
assets/bike/real_bike/source/ uploaded bike exports (OBJ, textures and USDZ)
src/k1bike/           our code: paths, scene builder (K1 + bike), robots/k1.py, tasks/ per phase
configs/              how to configure runs and W&B conventions
scripts/              setup_booster.sh, view_model.py, train.py, play.py
cluster/              GPU cluster notes and SLURM job template
tests/                quick checks that the models load and simulate (run in CI)
docs/decisions.md     decision log (what we chose and why)
docs/commands.md      command reference, separated by OS and execution target
docs/bike-dimensions.md physical bike measurements, estimates, and missing dimensions
AGENTS.md            shared instructions for coding agents
handover docs/        handover notes; credentials.md and PDFs are gitignored
```
### Bike model

View the models with the [Linux or macOS viewer commands](docs/commands.md#view-the-bike-or-the-k1-and-bike).
The viewer defaults to the green USDZ-derived bike at 56.5 cm wheelbase; its
physical parameters remain provisional (see [dimensions](docs/bike-dimensions.md)).
Use `--model real_bike` for the uploaded textured blue bike, or `--model primitive`
for the original shapes. All three support rolling wheels and steering. The
[scan notes](assets/bike/real_bike/README.md) explain its approximate part
boundaries and its saddle/handlebar height differences from the measured bike.
The `k1_bike` scene places the robot next to the bike; seated interaction is not implemented yet.

### Robot access

Ask a teammate for the robot login (user, IP, password). Never commit it.

The robot firmware must be v1.7.2 or newer before it can run our own policies
(check with `cat /opt/booster/version.txt` after SSH, on the robot's Linux system).
See [robot access commands](docs/commands.md#robot-access) for the distinction
between Linux-only SDK commands and SSH from either host OS.

### Team scripts (from PR #5)

- `scripts/robot_info.py`: connects to the real K1 with `boosteros` and prints its model, mode and joint count. Run from a Linux environment on the robot's network; see the command reference.
- `scripts/run_sample_task.sh`: Linux/macOS wrapper for Booster's pretrained MuJoCo tasks; defaults to `k1_fight` and accepts a task name plus deployment options.
- `scripts/connect-booster-k1.sh`, `scripts/mujoco-script.sh`: legacy setup experiments. Use `scripts/setup_booster.sh` and the command reference for the current pinned workflow.
- The ROS installer and Docker rebuild scripts under `scripts/` belong to an optional Ubuntu-container experiment. They are separate from the current mjlab workflow; see [legacy scripts](docs/commands.md#legacy-scripts).

### Troubleshooting

- See [platform-specific troubleshooting](docs/commands.md#platform-specific-troubleshooting)
  for the Linux GTK/Wayland workaround, Linux headless rendering, and the team's
  macOS hidden-file workaround.
