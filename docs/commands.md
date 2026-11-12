# Platform command reference

Run commands from the repository root unless a block explicitly changes directory.
The project uses Python 3.12 (`.python-version` and `pyproject.toml`).

The repository root is the directory containing this project's `pyproject.toml`;
its name is arbitrary. Booster checkouts normally live directly inside it as
`booster_assets/` and `booster_deploy/`. Setup also reuses the older
`booster/booster_assets/` and `booster/booster_deploy/` layout, creating top-level
relative symlinks for uv and the examples below. It does not move or duplicate
existing clones. Run setup before `uv sync` if
`booster_assets/` is absent.

| Execution target | Native MuJoCo viewer | Headless Python | Intended use |
|---|---|---|---|
| Linux host | `uv run python` | `uv run python` | Local viewing, tests, CPU smoke runs; GPU training with NVIDIA hardware |
| macOS host | `uv run mjpython` | `uv run python` | Local viewing, tests, CPU smoke runs |
| Linux GPU cluster | Use headless jobs | `uv run python` | NVIDIA GPU training through SLURM |
| Robot (Linux) | Not needed for deployment | Robot's configured Python environment | Hardware access and deployment |

`mjpython` is installed with MuJoCo. It is needed on macOS for the native passive
viewer, not for every Python command. See the [MuJoCo viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

## Setup

### Linux and macOS: install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal if `uv` is not yet on `PATH`.

### Linux and macOS: local bike viewer only

From the repository root, install the pinned asset checkout and base dependencies:

```bash
bash scripts/setup_booster.sh assets
uv sync
```

This local desktop workflow uses Python 3.12 and needs no GPU or optional
dependency group. The generated bike assets are included in the repository.
Continue with the [viewer commands](#view-the-bike-or-the-k1-and-bike), or follow
the [README walkthrough](../README.md#quick-start-watch-the-real-bike-roll-and-steer).
The fuller setup below adds pretrained-policy and training dependencies.

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

On Apple Silicon, use an arm64 Python. If uv selected an Intel interpreter, an
example using a Homebrew-installed Python 3.12 is:

```bash
# macOS / Apple Silicon only; requires Python at this path.
uv venv --python /opt/homebrew/bin/python3.12
uv sync --group sim --group train
```

## View the bike or the K1 and bike

The viewer defaults to the green, USDZ-derived bike scaled to **56.5 cm between
wheel centres**, with its stem extended to **45 cm grip-centre height**.
It loads `assets/bike/sketchfab_mujoco/bike.xml`, which contains
the separated meshes and provisional physics; it does not load the USDZ directly.
Use `--model primitive` to compare with the original blue prototype.
Use `--model real_bike` for the uploaded textured blue bike scan. Its uniformly
scaled saddle and grip heights differ from the measured bike; see the
[scan notes](../assets/bike/real_bike/README.md). The default remains green.

### Linux host

```bash
uv run python scripts/view_model.py bike
uv run python scripts/view_model.py bike --paused
uv run python scripts/view_model.py bike --model primitive
uv run python scripts/view_model.py bike --model real_bike --paused
uv run python scripts/view_model.py bike --model real_bike --push 1.5 --steer 12 --paused
uv run python scripts/view_model.py bike --push 3
uv run python scripts/view_model.py bike --lean 5 --drop 0.03
uv run python scripts/view_model.py bike --drop 0.3
uv run python scripts/view_model.py k1_bike
uv run python scripts/view_model.py k1_bike --model real_bike
```

### macOS host

```bash
uv run mjpython scripts/view_model.py bike
uv run mjpython scripts/view_model.py bike --paused
uv run mjpython scripts/view_model.py bike --model primitive
uv run mjpython scripts/view_model.py bike --model real_bike --paused
uv run mjpython scripts/view_model.py bike --model real_bike --push 1.5 --steer 12 --paused
uv run mjpython scripts/view_model.py bike --push 3
uv run mjpython scripts/view_model.py bike --lean 5 --drop 0.03
uv run mjpython scripts/view_model.py bike --drop 0.3
uv run mjpython scripts/view_model.py k1_bike
uv run mjpython scripts/view_model.py k1_bike --model real_bike
```

Run one viewer command at a time. `--push 3` sets an initial rolling speed of
3 m/s, including wheel rotation. `--lean` sets the initial roll angle in degrees;
`--drop` raises the bike by the specified metres before release. A 5° lean
demonstrates toppling, and a 0.3 m drop demonstrates gravity. These options only
change the bike's initial state, including in the combined scene.
`--steer 12` starts with 12° steering and leaves it passive after release; it
does not hold the angle or add a controller. Use `--paused` to inspect the
initial pose; **Space** pauses or resumes simulation. **R** resets the bike to
the requested initial state so you can repeat the push or drop. The camera
follows the bike, and the viewer prints the bike's configured mass at startup.

`k1_bike` places the K1 next to the bike; it does not seat or attach it. See the
[bike model notes](../assets/bike/README.md) for physics limitations and the
[physical measurements](bike-dimensions.md) before changing its parameters.

### Linux and macOS: regenerate the converted bike

This is headless preprocessing; ordinary Python is used on both platforms. It
preserves the original USDZ and writes the derivative under `assets/bike/sketchfab_mujoco/`:

```bash
uv run --with usd-core python scripts/convert_bike_usdz.py
```

The owner's **56.5 cm wheelbase** and **45 cm grip-centre height** are the defaults;
no dimension arguments are required. The default **1.5 kg** mass is a provisional
value within the owner's estimate, not a weighing. The height adjustment extends
the stem along the steering axis. Optional flags override these defaults.
Regeneration also prints the derived dimensions; check them against
[docs/bike-dimensions.md](bike-dimensions.md). The extra `usd-core` package is
not needed for normal viewing or tests.

### Linux and macOS: regenerate the real bike

From the repository root, headless with ordinary Python on both platforms:

```bash
uv run python scripts/convert_real_bike.py
```

This reads the exports in `assets/bike/real_bike/source/` and writes the derivative
under `assets/bike/real_bike/`. It uses the source colour texture through a
relative path; keep the source folder with the generated asset. No extra dependency
group is needed. Defaults are 0.565 m axle spacing and a provisional 1.5 kg mass;
`--wheelbase` and `--mass` override them. The scan's saddle and handlebars retain
their uniformly scaled geometry. Conversion details and limitations are in the
[scan notes](../assets/bike/real_bike/README.md).

## Run Booster's pretrained policy

These examples enter the pinned `booster_deploy` checkout. Return to the project
root before running commands from other sections.

### Linux host

```bash
cd booster_deploy
uv run --group sim python scripts/deploy.py --task k1_walk --mujoco
```

### macOS host

```bash
cd booster_deploy
uv run --group sim mjpython scripts/deploy.py --task k1_walk --mujoco
```

From the repository root, the wrapper selects the launcher automatically:

```bash
# Linux and macOS; defaults to k1_fight if no task is supplied.
bash scripts/run_sample_task.sh k1_walk
```

## Train our mjlab task

New runs train with `scripts/train.py` on the Linux GPU cluster. Walk runs 1 and 2 were
trained on Google Colab; their notebooks are kept as records in `phases/walk/runs/runN/`.

### Linux GPU cluster node or NVIDIA host

Run on an allocated GPU node (or a machine where you are allowed to use the GPU):

```bash
uv run --group train python scripts/train.py LAAI-K1-Walk-Flat \
  --env.scene.num-envs 4096 --agent.run-name walk-baseline-ep
```

### Linux and macOS: CPU smoke run

This small run checks the training pipeline; use the Linux GPU workflow for full
training. It is headless, so macOS also uses ordinary Python:

```bash
uv run --group train python scripts/train.py LAAI-K1-Walk-Flat \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 \
  --agent.logger tensorboard
```

## Watch our trained policy

Replace the example checkpoint path with one from your run. These examples use
one environment on CPU and explicitly select the native viewer.

### Linux host

```bash
uv run --group train python scripts/play.py LAAI-K1-Walk-Flat \
  --checkpoint-file logs/rsl_rl/laai_walk/YOUR_RUN/model_1000.pt \
  --device cpu --num-envs 1 --viewer native
```

### macOS host

```bash
uv run --group train mjpython scripts/play.py LAAI-K1-Walk-Flat \
  --checkpoint-file logs/rsl_rl/laai_walk/YOUR_RUN/model_1000.pt \
  --device cpu --num-envs 1 --viewer native
```

For mjlab's browser viewer, both platforms use `uv run --group train python`
with `--viewer viser` instead. This is separate from Booster's `deploy.py` viewer.

## Export our walk policy for booster_deploy

### Linux GPU cluster or NVIDIA host

Writes `policy.pt`, checks it against mjlab's own actions (must print `OK`) and
optionally records a video. `MUJOCO_GL=egl` is for headless Linux NVIDIA rendering:

```bash
MUJOCO_GL=egl uv run --group train python -m phases.walk.export \
  --checkpoint logs/rsl_rl/laai_walk/YOUR_RUN/model_3999.pt --video walk.mp4
```

Keep the result as `phases/walk/runs/runN/policy.pt` (see [phases/README.md](../phases/README.md)).

## Run our walk policy in Booster's MuJoCo

Install the walk phase's task file, the dry-run script and a policy into the `booster_deploy`
checkout (see [deploy/README.md](../deploy/README.md)), then use Booster's `deploy.py`
with task `laai_walk` or `laai_walk_ankle_kd15`. Type speed commands in the
terminal as `x y yaw`, for example `0.3 0 0`.

### Linux and macOS

```bash
bash deploy/install.sh walk                                    # policy of the newest walk run
bash deploy/install.sh walk phases/walk/runs/run1/policy.pt    # a specific policy
```

### Linux host

```bash
cd booster_deploy
uv run --group sim python scripts/deploy.py --task laai_walk_ankle_kd15 --mujoco
```

### macOS host

```bash
cd booster_deploy
uv run --group sim mjpython scripts/deploy.py --task laai_walk_ankle_kd15 --mujoco
```

Rehearse the robot dry run without a robot (headless, Linux and macOS):

```bash
cd booster_deploy
uv run --group sim python scripts/dry_run.py --task laai_walk_ankle_kd15 --source mujoco --noise
```

## Robot test

### Linux and macOS host: copy to the robot

Installs our files into booster_deploy, copies it to the robot and installs its Python
packages there. You type the robot password once. On macOS, use the Terminal app: editors
without Local Network permission cannot reach the robot.

```bash
bash deploy/setup_robot.sh ROBOT_IP walk
```

### Robot (Linux): dry run, then the policy

After `ssh booster@ROBOT_IP`, these commands run on the robot's Linux system. First a dry
run (no motor commands; robot standing in Booster's walking mode):

```bash
cd ~/booster_deploy && source .venv/bin/activate
source /opt/booster/BoosterRos2Interface/install/setup.bash
python3 scripts/dry_run.py --task laai_walk_ankle_kd15 --seconds 20 | tee logs/dry_run_report_$(date +%Y%m%d_%H%M%S).txt
```

Then the policy, with someone ready to catch the robot. Add `--exit-mode damping` only
with a harness (the robot goes limp after a stop):

```bash
python3 scripts/deploy.py --task laai_walk_ankle_kd15
```

| Key / remote | Action |
|---|---|
| `x` / X | Custom mode: Booster's own policy stands the robot |
| `r` / A | Start our policy (0.5 s handover) |
| `w` `s` / left stick | Forward / back (`w`: +0.05 m/s per press) |
| `a` `d` / left stick | Sideways (+0.02 m/s per press) |
| `q` `e` / right stick | Turn (+0.06 rad/s per press) |
| Space | Speeds to zero; the policy keeps running |
| Ctrl+C | Stop; the robot enters the exit mode |

Each safety stop prints the joint and saves the last 5 s to `logs/stop_<robot time>.npz`.
The robot clock runs 7 hours ahead of Amsterdam time.

### Linux and macOS host: keep the robot logs with the run

```bash
scp 'booster@ROBOT_IP:~/booster_deploy/logs/stop_*' phases/walk/runs/runN/robot_logs/
scp 'booster@ROBOT_IP:~/booster_deploy/logs/dry_run_*' phases/walk/runs/runN/dry_run/
```

## Development checks

### Linux and macOS

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Robot access

### Linux host on the robot network

The `boosteros` SDK dependency group is Linux-only:

```bash
uv run --group robot python scripts/robot_info.py
```

### Linux and macOS: SSH client

Ask a teammate for the robot user, IP and password; never commit them:

```bash
ssh ROBOT_USER@ROBOT_IP
```

After SSH, commands run on the robot's Linux system. Follow the deployment
instructions for that environment; the laptop's uv environment is not transferred
by SSH. See the [Booster deployment instructions](https://github.com/BoosterRobotics/booster_deploy#run-sim2real-real-robots).

## Platform-specific troubleshooting

### Linux desktop: GTK/Wayland workaround

If the viewer fails with the GNOME `antialiasing` schema error, try the X11 backend
for that invocation only:

```bash
GDK_BACKEND=x11 uv run python scripts/view_model.py bike
```

### Linux NVIDIA machine: headless rendering

Set `MUJOCO_GL=egl` for rollout videos on a headless Linux NVIDIA machine or
cluster node. This setting is not for launching a local native viewer.

### macOS: hidden environment files

For the team's documented macOS hidden-file issue, the platform-specific command is:

```bash
chflags -R nohidden .venv
```

## Legacy scripts

`scripts/connect-booster-k1.sh` and `scripts/mujoco-script.sh` are earlier setup
experiments. Use the pinned setup and platform-specific commands above for the
current workflow.

`scripts/docker-install-ros2-humble-ubuntu-22-04.sh` runs inside an Ubuntu 22.04
Linux container. Its old bridge-launch section is not part of the current mjlab
workflow and has not been validated. `scripts/rebuild-ros2-cont.sh` operates on
that optional Docker setup; it is separate from local simulation and GPU training.
