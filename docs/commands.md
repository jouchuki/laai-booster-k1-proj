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
| Linux GPU cluster | Use headless jobs | `uv run python` | CPU or NVIDIA GPU training through SLURM |
| Robot (Linux) | Not needed for deployment | Robot's configured Python environment | Hardware access and deployment |

`mjpython` is installed with MuJoCo. It is needed on macOS for the native passive
viewer, not for every Python command. See the [MuJoCo viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

On **Hex (Linux GPU cluster)**, run simulation, training, playback, export and tests in
SLURM allocations, including CPU smoke runs. Use the login node for setup, submission,
monitoring and file transfers. Follow [Hex setup and rules](../cluster/README.md) for the
Miniconda Python 3.12 interpreter, locked uv dependencies, storage quotas and interactive
allocations. Desktop viewer examples below are for local hosts.

Training uses NumPy 2; Booster's robot SDK requires NumPy below 2. The `train` and `robot`
dependency groups are declared incompatible in uv and must use separate environments.
Install the groups needed for the workflow rather than using `--all-groups`.

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
the [README walkthrough](../README.md#quick-start-watch-the-bike-scan-roll-and-steer).
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

The viewer defaults to the real bike, `models/puky_bike` (the textured scan,
calibrated to its measurements). Use `--model yellow_bike` for the earlier green bike,
made from a USDZ model from the internet (Sketchfab) (**56.5 cm between wheel centres**, **45 cm grip-centre height**),
and `--model primitive_bike` to compare with the original blue prototype.
The real bike is the uploaded textured blue scan, rescaled to 50 cm wheelbase, 22 cm
tyre diameter, 42 cm saddle top and 53 cm grip centres; see the
[real bike notes](../models/puky_bike/README.md).
Earlier names still work as aliases: `bike_scan` selects `puky_bike`; `bike` and
`sketchfab` select `yellow_bike`; `bike_primitive` and `primitive` select `primitive_bike`.

### Linux host

```bash
uv run python scripts/view_model.py bike
uv run python scripts/view_model.py bike --paused
uv run python scripts/view_model.py bike --model yellow_bike
uv run python scripts/view_model.py bike --model primitive_bike
uv run python scripts/view_model.py bike --push 1.5 --steer 12 --paused
uv run python scripts/view_model.py bike --push 3
uv run python scripts/view_model.py bike --lean 5 --drop 0.03
uv run python scripts/view_model.py bike --drop 0.3
uv run python scripts/view_model.py k1_bike
uv run python scripts/view_model.py k1_bike --model yellow_bike
uv run python scripts/view_model.py k1_seated
uv run python scripts/view_model.py k1_seated --model yellow_bike
```

### macOS host

```bash
uv run mjpython scripts/view_model.py bike
uv run mjpython scripts/view_model.py bike --paused
uv run mjpython scripts/view_model.py bike --model yellow_bike
uv run mjpython scripts/view_model.py bike --model primitive_bike
uv run mjpython scripts/view_model.py bike --push 1.5 --steer 12 --paused
uv run mjpython scripts/view_model.py bike --push 3
uv run mjpython scripts/view_model.py bike --lean 5 --drop 0.03
uv run mjpython scripts/view_model.py bike --drop 0.3
uv run mjpython scripts/view_model.py k1_bike
uv run mjpython scripts/view_model.py k1_bike --model yellow_bike
uv run mjpython scripts/view_model.py k1_seated
uv run mjpython scripts/view_model.py k1_seated --model yellow_bike
```

Run one viewer command at a time. `--push 3` sets an initial rolling speed of
3 m/s, including wheel rotation. `--lean` sets the initial roll angle in degrees;
`--drop` raises the bike by the specified metres before release. A 5° lean
demonstrates toppling, and a 0.3 m drop demonstrates gravity. These options only
change the bike's initial state in the bike-only and unattached combined scenes.
`--steer 12` starts with 12° steering and leaves it passive after release; it
does not hold the angle or add a controller. `k1_seated` rejects `--push`,
`--steer`, `--lean` and `--drop` changes because its pose and hand attachments
are solved for the nominal bike state.
Use `--paused` to inspect the initial pose; **Space** pauses or resumes simulation.
**R** restores the requested initial state, including the seated keyframe and
pose-holding target in `k1_seated`. The camera follows the bike, and the viewer
prints the bike's configured mass at startup.

`k1_bike` places the K1 next to the bike, unattached. `k1_seated` starts the K1
sitting on the chosen bike with both hands welded to the grips, and holds that pose
with the k1_walk PD gains (see `src/k1bike/seat.py`). It works for every bike with a
solved pose in `models/k1_on_bike/` (now `puky_bike` and `yellow_bike`); for any other bike it
prints the command that solves one. See the
[bike model notes](../models/README.md) for physics limitations and the
[physical measurements](bike-dimensions.md) before changing its parameters.

### Linux and macOS: re-solve the K1's seated pose

Rerun after changing a bike model or the clip geometry in `src/k1bike/seat.py`
(headless; needs the `sim` group for scipy). It writes `models/k1_on_bike/<bike>.json`
and refuses to write a pose that does not fit (it prints which target failed):

```bash
uv run --group sim python scripts/solve_seated_pose.py                 # the real bike (puky_bike)
uv run --group sim python scripts/solve_seated_pose.py --bike yellow_bike     # any other bike
```

A new bike only needs the standard part names (`frame`, `fork`, `saddle`, `seat`,
`left_grip`, `right_grip`, `steer`, `root`), an entry in `BIKES` in `src/k1bike/paths.py`,
and a solved pose.

### Linux and macOS: regenerate the converted bike

This is headless preprocessing; ordinary Python is used on both platforms. It
reads the original USDZ in `models/yellow_bike/source/` and writes the derivative
(`bike.xml`, `meshes/`, `measurements.json`) into `models/yellow_bike/`:

```bash
uv run --with usd-core python scripts/convert_yellow_bike.py
```

The owner's **56.5 cm wheelbase** and **45 cm grip-centre height** are the defaults;
no dimension arguments are required. The default **1.5 kg** mass is a provisional
value within the owner's estimate, not a weighing. The height adjustment extends
the stem along the steering axis. Optional flags override these defaults.
Regeneration also prints the derived dimensions; check them against
[docs/bike-dimensions.md](bike-dimensions.md). The extra `usd-core` package is
not needed for normal viewing or tests.

### Linux and macOS: regenerate the bike scan

From the repository root, headless with ordinary Python on both platforms:

```bash
uv run python scripts/convert_puky_bike.py
```

This reads the exports in `models/puky_bike/source/` and writes the derivative
under `models/puky_bike/`. It uses the source colour texture through a
relative path; keep the source folder with the generated asset. No extra dependency
group is needed. Defaults are 0.50 m axle spacing, 0.22 m tyre diameter,
0.42 m saddle-top height, 0.53 m grip-centre height and 3.7 kg mass (the midpoint
of the owner's reported 3.5–3.9 kg range for `puky_bike`). Optional `--wheelbase`,
`--tyre-diameter`, `--saddle-height` and `--handlebar-height` set dimensions
independently in metres; `--mass` sets kilograms. Wheelbase determines the
uniform scale; radial wheel sizing and local post adjustments fit the other
targets. See [the measurements](bike-dimensions.md#measurements-for-puky_bike-only-2026-11-12).
The rescaling process and limitations are in the
[real bike notes](../models/puky_bike/README.md).

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
trained on Google Colab; their notebooks are kept as records in `experiments/walk/runs/runN/`.
To submit a training job with SLURM, see [cluster/README.md](../cluster/README.md).
Checkpoints, TensorBoard files and videos go to `logs/<experiment>/<date>_<run-name>/`
(gitignored), for example `logs/laai_walk/2026-11-12_10-15-00_walk-baseline-ep/`.

Runs are tracked in Weights & Biases, project `laai-k1`. Each experiment is a group
(`laai_walk`, `laai_sit`) and each run is tagged with its task id. W&B records the curves, every
reward term (`Episode_Reward/*`), termination causes, tracking metrics, the full config, the
git diff, checkpoints and videos. Log in once per machine:

```bash
# Linux and macOS
uv run --group train wandb login
```

Alternatively, keep `WANDB_API_KEY` in the repository root's gitignored `.env`. Hex's batch
script loads that file automatically when present in the **Hex checkout**; it is not copied
by Git. For a direct training command on a **Linux NVIDIA host or allocated GPU node**, from
the repository root, add uv's `--env-file .env` option:

```bash
uv run --env-file .env --group train python scripts/train.py LAAI-K1-Sit-Puky \
  --env.scene.num-envs 4096 --agent.run-name stage0
```

uv reads the file as environment data; do not source it as a shell script or commit it.

Without an API key, `scripts/train.py` logs offline to `logs/wandb/` and prints the
`wandb sync` command that uploads the run later. Add `--agent.logger tensorboard` to skip W&B.

### Linux GPU cluster allocated node or NVIDIA host

Run on an allocated GPU node (or a machine where you are allowed to use the GPU):

```bash
uv run --group train python scripts/train.py LAAI-K1-Walk-Flat \
  --env.scene.num-envs 4096 --agent.run-name walk-baseline-ep
```

### Linux and macOS: CPU smoke run

This small run checks the training pipeline. CPU training with more environments is also
possible; GPUs improve parallel simulation throughput. This is headless, so macOS also uses
ordinary Python:

```bash
uv run --group train python scripts/train.py LAAI-K1-Walk-Flat \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 \
  --agent.logger tensorboard
```

### Linux and macOS: summarize a finished run

Prints the "Simulation results" block for the run's `notes.md` and a row for the experiment's
Runs table. Each number is the mean over the last 50 iterations (`--last N` changes this). It
also prints the W&B link, or the `wandb sync` command for an offline run. It reads the run's
TensorBoard files, so it works offline:

```bash
uv run --group train python scripts/run_summary.py logs/laai_walk/YOUR_RUN
```

## Watch our trained policy

Replace the example checkpoint path with one from your run. These examples use
one environment on CPU and explicitly select the native viewer.

### Linux host

```bash
uv run --group train python scripts/play.py LAAI-K1-Walk-Flat \
  --checkpoint-file logs/laai_walk/YOUR_RUN/model_1000.pt \
  --device cpu --num-envs 1 --viewer native
```

### macOS host

```bash
uv run --group train mjpython scripts/play.py LAAI-K1-Walk-Flat \
  --checkpoint-file logs/laai_walk/YOUR_RUN/model_1000.pt \
  --device cpu --num-envs 1 --viewer native
```

For mjlab's browser viewer, both platforms use `uv run --group train python`
with `--viewer viser` instead. This is separate from Booster's `deploy.py` viewer.

## Export our walk policy for booster_deploy

### Linux GPU cluster allocated node or NVIDIA host

Writes `policy.pt`, checks it against mjlab's own actions (must print `OK`) and
optionally records a video. `MUJOCO_GL=egl` is for headless Linux NVIDIA rendering:

```bash
MUJOCO_GL=egl uv run --group train python -m experiments.walk.export \
  --checkpoint logs/laai_walk/YOUR_RUN/model_3999.pt --video walk.mp4
```

Keep the result as `experiments/walk/runs/runN/policy.pt` (see [experiments/README.md](../experiments/README.md)).

## Run our walk policy in Booster's MuJoCo

Install the walk phase's task file, the dry-run script and a policy into the `booster_deploy`
checkout (see [robot/README.md](../robot/README.md)), then use Booster's `deploy.py`
with task `laai_walk` or `laai_walk_ankle_kd15`. Type speed commands in the
terminal as `x y yaw`, for example `0.3 0 0`.

### Linux and macOS

```bash
bash robot/install.sh walk                                    # policy of the newest walk run
bash robot/install.sh walk experiments/walk/runs/run1/policy.pt    # a specific policy
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
bash robot/setup_robot.sh ROBOT_IP walk
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
scp 'booster@ROBOT_IP:~/booster_deploy/logs/stop_*' experiments/walk/runs/runN/robot_logs/
scp 'booster@ROBOT_IP:~/booster_deploy/logs/dry_run_*' experiments/walk/runs/runN/dry_run/
```

## Train the sit task (phase 3)

`LAAI-K1-Sit-Puky`: the K1 seated on the Puky, bike held upright (see
[experiments/sit/README.md](../experiments/sit/README.md)). Logs go to `logs/laai_sit/`.

### Linux GPU cluster: submit on Hex

From the repository root on the login node after the
[cluster setup](../cluster/README.md#first-time-setup-on-the-cluster), submit a short GPU
smoke job first:

```bash
mkdir -p logs
sbatch --job-name=sit-smoke --time=00:30:00 cluster/train.slurm LAAI-K1-Sit-Puky \
  --env.scene.num-envs 32 --agent.max-iterations 3 --agent.run-name smoke \
  --agent.logger tensorboard
```

After the smoke job completes successfully, submit stage 0:

```bash
sbatch --job-name=sit-stage0 cluster/train.slurm LAAI-K1-Sit-Puky \
  --agent.run-name stage0
```

After stage 0 finishes, replace the checkpoint path with its actual file and submit stage 1:

```bash
sbatch --job-name=sit-stage1 cluster/train.slurm LAAI-K1-Sit-Puky-Lean \
  --init-from logs/laai_sit/YOUR_STAGE0_RUN/model_1499.pt --agent.run-name stage1
```

The shared script requests the student `batch-gpu` partition, one RTX 4500 GPU, five CPUs,
32 GB RAM and eight hours. See [cluster/README.md](../cluster/README.md) for monitoring and
GPU/driver requirements. Training dependencies must be installed before submission.

### Linux GPU cluster: CPU-only training on Hex

From the repository root on the login node, after the same Miniconda and uv setup, submit
a CPU smoke test:

```bash
mkdir -p logs
sbatch --job-name=sit-cpu-smoke --time=00:15:00 cluster/train_cpu.slurm LAAI-K1-Sit-Puky \
  --env.scene.num-envs 8 --agent.max-iterations 3 --agent.run-name cpu-smoke \
  --agent.logger wandb
```

After it passes, train stage 0 with 256 environments and 1,500 iterations:

```bash
sbatch --job-name=sit-cpu-stage0 cluster/train_cpu.slurm LAAI-K1-Sit-Puky \
  --agent.run-name cpu-stage0
```

The CPU launcher requests `batch`, 4 CPUs, 4 GB RAM and 8 hours, forces `--gpu-ids None`,
and accepts the same checkpoint and training options as the GPU launcher. See the
[CPU cluster guide](../cluster/README.md#submit-cpu-sit-training) for stage 1 and timing limits.

### Linux GPU cluster allocated node or NVIDIA host

```bash
uv run --group train python scripts/train.py LAAI-K1-Sit-Puky \
  --env.scene.num-envs 4096 --agent.run-name run1
```

Stage 1 (bike leans on a softening stand), started from a stage 0 checkpoint. `--init-from`
continues from that checkpoint's weights in a new run folder; it must be in `logs/laai_sit/`:

```bash
uv run --group train python scripts/train.py LAAI-K1-Sit-Puky-Lean \
  --init-from logs/laai_sit/YOUR_STAGE0_RUN/model_1499.pt \
  --env.scene.num-envs 4096 --agent.run-name stage1
```

### Linux and macOS: CPU smoke run

```bash
uv run --group train python scripts/train.py LAAI-K1-Sit-Puky \
  --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 --agent.logger tensorboard
```

### Linux host: watch a checkpoint

```bash
uv run --group train python scripts/play.py LAAI-K1-Sit-Puky \
  --checkpoint-file logs/laai_sit/YOUR_RUN/model_1499.pt --device cpu --num-envs 1 --viewer native
```

### macOS host: watch a checkpoint

```bash
uv run --group train mjpython scripts/play.py LAAI-K1-Sit-Puky \
  --checkpoint-file logs/laai_sit/YOUR_RUN/model_1499.pt --device cpu --num-envs 1 --viewer native
```

### Linux GPU cluster allocated node or NVIDIA host: export for booster_deploy

```bash
MUJOCO_GL=egl uv run --group train python -m experiments.sit.export \
  --checkpoint logs/laai_sit/YOUR_RUN/model_1499.pt --video sit.mp4
```

The export must print `Deploy-format check: ... (OK)`. Keep the result as
`experiments/sit/runs/runN/policy.pt`; the booster_deploy task `laai_sit` runs it on the robot
([next section](#robot-test-the-trained-sit-policy-phase-3)).

## Robot test: the trained sit policy (phase 3)

`laai_sit` runs a trained policy around the seated pose, with the same safety stops as
`laai_sit_hold` plus the walk task's shake watchdog. Same placement, people and keys as
[the hold](#robot-test-hold-the-seated-pose-phase-3); set the saddle as in
[the 2026-11-19 robot test](../experiments/sit/robot_tests/2026-11-19/notes.md). Keep the bike
held for the first tests.

It runs any policy exported in this format: a stage 0 run (`experiments/sit/runs/runN/policy.pt`,
the default) or a free-bike run (`experiments/sit_free/runs/runN/policy.pt`, given with
`--policy` and after `sit`). There is no stage 0 run in the repository yet. Of the free-bike
runs, **run 2** sits calmly in the rehearsal (no stop up to a 40 N push); run 1 trips the
hip-roll torque stop within a second, so do not take it to the robot.

### Linux and macOS: rehearse it in simulation on the bike

Free-bike run 2, with the bike held (as on the robot):

```bash
uv run --group sim python experiments/sit/check_hold.py --task laai_sit --bike-held --policy experiments/sit_free/runs/run2/policy.pt
```

The newest stage 0 run's `policy.pt`, with the bike held, then released:

```bash
uv run --group sim python experiments/sit/check_hold.py --task laai_sit --bike-held
```

```bash
uv run --group sim python experiments/sit/check_hold.py --task laai_sit
```

A specific policy, with a sideways push halfway through:

```bash
uv run --group sim python experiments/sit/check_hold.py --task laai_sit --policy experiments/sit/runs/run1/policy.pt --push 40
```

### Linux and macOS host: copy to the robot

Installs the given policy as `models/laai_sit.pt`; without one, the newest stage 0 run's `policy.pt`:

```bash
bash robot/setup_robot.sh ROBOT_IP sit experiments/sit_free/runs/run2/policy.pt
```

```bash
bash robot/setup_robot.sh ROBOT_IP sit
```

### Robot (Linux): dry run with the robot placed on the bike, then the policy

```bash
cd ~/booster_deploy && source .venv/bin/activate
source /opt/booster/BoosterRos2Interface/install/setup.bash
python3 scripts/dry_run.py --task laai_sit --seconds 10 | tee logs/dry_run_report_$(date +%Y%m%d_%H%M%S).txt
```

```bash
python3 scripts/deploy.py --task laai_sit_ankle_kd15
```

X slides the joints to the seated pose in 1 s, then press `r` (A) within a few seconds: the
safety stops run only from there. Ctrl+C stops into damping (limp) mode. Never add
`--exit-mode walking`.

`laai_sit_ankle_kd15` is `laai_sit` with ankle damping 1.5 instead of 1.0. Use it: with 1.0
the ankle rolls oscillated after a push in the first policy test
([2026-11-19](../experiments/sit/robot_tests/2026-11-19/notes.md#policy-test-free-bike-run-2)).
Plain `laai_sit` still works (`--task laai_sit`, for the dry run too).

Rehearse a robot placement before the policy runs: `--start-from` takes a dry run's log
(copied to the laptop) and starts the rehearsal in that placement:

```bash
uv run --group sim python experiments/sit/check_hold.py --task laai_sit_ankle_kd15 --bike-held --policy experiments/sit_free/runs/run2/policy.pt --start-from experiments/sit/robot_tests/YYYY-MM-DD/dry_run_laai_sit_ros_TIME.npz
```

## Compare free-bike training on Hex

Task `LAAI-K1-Sit-Puky-Free` removes the fixed root and spring stand. Its separate experiment
is [experiments/sit_free/](../experiments/sit_free/README.md), with raw output under
`logs/laai_sit_free/`. Both comparison jobs use W&B, 256 environments, seed 42 and 1,500 new
iterations. One imports only the pretrained actor; both reset the critic, optimizer and
counters. Do not use full-state `--init-from` for this comparison.

**Linux GPU cluster (Hex login node)**, from the repository root after the documented
Miniconda/uv setup. Replace the checkpoint with a completed snapshot from the fixed-bike run.
Check storage usage first. Validate both scheduler requests without submitting jobs:

```bash
uv run --no-sync --group train python scripts/submit_sit_free.py \
  logs/laai_sit/YOUR_STAGE0_RUN/model_900.pt --test-only
```

Submit both CPU jobs:

```bash
uv run --no-sync --group train python scripts/submit_sit_free.py \
  logs/laai_sit/YOUR_STAGE0_RUN/model_900.pt
```

The helper requests the student `batch` CPU partition through `cluster/train_cpu.slurm`:
4 CPUs and 4 GB RAM per job, no GPU, and a 12-hour limit to allow for the free-body physics.
It freezes the checkpoint and source in `logs/laai_sit_free/comparisons/<comparison>/`,
records job IDs in `manifest.json`, and gives both runs one W&B comparison group.
Each final checkpoint triggers the same held-out evaluation inside its allocation;
results appear in `evaluation.json` and W&B `Evaluation/*`. Compare these along with
episode duration, fall causes, bike tilt and saddle support. The original run stays separate.

## Train the free bike with a push curriculum

`LAAI-K1-Sit-Puky-Free-Push` is the free-bike task with pushes growing from 0.2 to 2.0 m/s as
the policy keeps the bike up; why and how: [experiments/sit_free/README.md](../experiments/sit_free/README.md#push-curriculum-laai-k1-sit-puky-free-push).

### Linux GPU cluster: train from free-bike run 2's actor

```bash
sbatch --job-name=sit-free-push cluster/train.slurm LAAI-K1-Sit-Puky-Free-Push --agent.init-policy-from experiments/sit_free/runs/run2/actor_checkpoint.pt --agent.max-iterations 3000 --agent.run-name push-run3
```

### Linux and macOS: CPU smoke run

```bash
uv run --group train python scripts/train.py LAAI-K1-Sit-Puky-Free-Push --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 --agent.logger tensorboard --agent.init-policy-from experiments/sit_free/runs/run2/actor_checkpoint.pt
```

The job's `evaluation.json` includes `push_sweep`: success at 0.2-2.0 m/s for the policy and
for zero actions. The run has learned to balance where `beats_zero_action_at_mps` lists a push.

## Compare the trained free-bike policies locally

The completed comparison is archived in `experiments/sit_free/runs/`: both runs include
full training scalar histories, compressed console logs, configs, evaluation reports,
exported policies and compact actor checkpoints. See the
[artifact index and recording](../experiments/sit_free/runs/comparison/README.md).
The original full checkpoints remain in the Hex run folders and in the local
`logs/laai_sit_free/` download. The compact checkpoints contain the same actor tensors
and support playback without downloading optimizer or critic state.

**Linux host**, repository root: open one split-screen window with synchronized clocks,
camera angles, initial seed and pushes. The pretrained policy is on the left, scratch on
the right. Each side stops at its first failure or after 10 seconds.

```bash
uv run --group train python scripts/compare_sit_free.py \
  --pretrained experiments/sit_free/runs/run1/actor_checkpoint.pt \
  --scratch experiments/sit_free/runs/run2/actor_checkpoint.pt
```

Space pauses/resumes; Enter replays both; N selects the next shared seed; Escape closes.
Drag to rotate both cameras and use the wheel to zoom. Add `--paused` to start paused.
Add `--once --record logs/laai_sit_free/comparison-final.mp4` to record a paired trial
and close afterwards. This one-world visual trial is separate from the 32-world evaluation.
For the Linux GTK schema error described below, prefix this invocation with
`GDK_BACKEND=x11`; keep that workaround local to the command.

**Linux and macOS**, repository root, headless: export either checkpoint into the existing
Booster deployment observation/action layout. Replace `YOUR_RUN` with the actual run name.

```bash
uv run --group train python -m experiments.sit_free.export \
  --checkpoint logs/laai_sit_free/YOUR_RUN/model_1499.pt \
  --out logs/laai_sit_free/YOUR_RUN/policy.pt --device cpu
```

The export checks its actions against the checkpoint and prints `OK` when they match.
Export verification does not validate deployment on the real robot. The split-screen
viewer has been validated on Linux only.

## Robot test: hold the seated pose (phase 3)

`laai_sit_hold` holds the solved seated pose on the Puky. It does not balance. The full
procedure (people, placement, dry run, keys) is in [experiments/sit/README.md](../experiments/sit/README.md).

### Linux and macOS: rehearse it in simulation on the bike

```bash
uv run --group sim python experiments/sit/check_hold.py
```

### Linux and macOS host: copy to the robot

```bash
bash robot/setup_robot.sh ROBOT_IP sit
```

### Robot (Linux): dry run with the robot placed on the bike, then hold

```bash
cd ~/booster_deploy && source .venv/bin/activate
source /opt/booster/BoosterRos2Interface/install/setup.bash
python3 scripts/dry_run.py --task laai_sit_hold --seconds 10 | tee logs/dry_run_report_$(date +%Y%m%d_%H%M%S).txt
```

```bash
python3 scripts/deploy.py --task laai_sit_hold
```

X slides the joints to the seated pose in 1 s, A starts the hold, Ctrl+C stops into damping
(limp) mode. Never add `--exit-mode walking` to this task.

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
uv run --group robot python robot/robot_info.py
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

If the project folder syncs with iCloud, macOS can mark the environment's `.pth` files
hidden, and Python then skips them (`ModuleNotFoundError: No module named 'k1bike'`, or
`cannot import name 'BOOSTER_ASSETS_DIR'`). Clear the flag:

```bash
chflags -R nohidden .venv/
```

To stop it recurring, keep the environment out of iCloud: move it to `.venv.nosync`
(iCloud skips `.nosync` names) and link `.venv` to it.

```bash
mv .venv .venv.nosync && ln -s .venv.nosync .venv && chflags -R nohidden .venv.nosync
```

## Legacy ROS 2 Docker scripts

`legacy/ros2_docker/` holds an optional, unvalidated Ubuntu 22.04 container experiment
with ROS 2 Humble. It is not part of the simulation, training or robot workflow above;
see [its README](../legacy/ros2_docker/README.md).
