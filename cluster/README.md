# Hex cluster

This page's commands run on the **Linux GPU cluster**, from the repository root.
Linux and macOS can be the SSH client; after connecting, commands execute on Hex's Linux system.
For local Linux/macOS viewing and CPU smoke runs, use [docs/commands.md](../docs/commands.md).

## Cluster settings

The [CI group cluster guide](https://github.com/ci-group/slurm-cluster-doc#partitions)
documents the student GPU partition and its limit of five CPUs per GPU. The live scheduler
and this account's association were checked on 2026-11-17:

| Setting | Value |
|---|---|
| SSH alias | `hex`, through `bigmac` (configure your own account as in the guide) |
| Scheduler / student GPU partition | SLURM / `batch-gpu` |
| Account association observed | `students`, `student_qos`; use your own account's permissions |
| Script request | 1 RTX 4500 GPU (`gpu:rtx4500:1`), 5 CPUs, 32 GB RAM, 8 hours |
| CPU script request | `batch`, 1 node, 1 task, 4 CPUs, 4 GB RAM, 8 hours; no GPU |
| Partition time limit | 2 days |
| RTX 4500 node | `node08`, with 4 GPUs; availability changes with running jobs |
| Persistent storage | Home is shared across nodes; the guide also lists `/scratch/<username>` |
| Compute-node driver and internet | Not validated; check with the smoke job below |

The GPU type is deliberate: the partition also includes GTX 1070 GPUs, which cannot run
the CUDA 13 build of Warp in our lockfile. [Warp's installation guide](https://nvidia.github.io/warp/stable/user_guide/installation.html)
requires a Turing (`sm_75`) or newer GPU and an R580-series or newer driver for its current
PyPI build. The smoke job checks the installed driver with the real training stack. A
CUDA module does not upgrade the host driver.

SLURM options before the script path override its defaults. For example, `--time=00:30:00`
requests a shorter job. RTX 4500 availability can mean queueing; another GPU type should
only be selected after checking its driver, architecture and available GPU memory.
Keep requests within the two-day partition limit. This launcher requires one node, one task,
one GPU and 1-5 CPUs; it rejects allocations that differ. Memory and time overrides must
also pass the scheduler's account, QoS and partition checks. Use the shortest sufficient
runtime and release interactive allocations when finished. Staff GPU partitions (`gpu`,
`gpu-short`, `gpu-long`) require staff approval; our examples use `batch-gpu` throughout.

For this project, run simulation, training, playback, export and tests inside SLURM
allocations. Use the login node for setup, submission, monitoring and transfers. Keep SLURM's `CUDA_VISIBLE_DEVICES`
unchanged. The script limits OpenMP, MKL, OpenBLAS and NumExpr threads to the CPU allocation.

## Storage

The [storage guide](https://github.com/ci-group/slurm-cluster-doc#file-system) lists 200 GB
per user on `/scratch`. It gives conflicting home limits (50 and 60 GB); use 50 GB as the
conservative budget and confirm your actual quota with staff. Its seven-day over-quota
grace period is for reducing usage, not extra storage capacity.

Use your own shared `/scratch/<username>` directory for the checkout, environments, caches
and training output. Check personal usage and filesystem free space before setup and new
runs. **Linux GPU cluster**, from any directory:

```bash
du -sh "$HOME" "/scratch/$USER"
df -h "$HOME" "/scratch/$USER"
```

These show usage and free space, not your enforced quota; `quota` was unavailable in the
checked account. Confirm limits with staff if needed. Keep checkpoints, videos and W&B
files within your budget; archive required results before removing your own obsolete runs
or caches. The repository does not automatically delete files. Shared free space is not a
personal allowance.

Job-local `/tmp` data disappears when the job ends. If you use it for intermediate files,
copy every result you need to persistent storage before exit. The default job keeps logs in
the shared checkout; do not move its logs or resume checkpoints into `/tmp`.

## Connect

**Linux and macOS host**, from any directory, using your existing SSH configuration:

```bash
ssh hex
```

The following sections run on the **Linux GPU cluster**. Use the repository root as the
working directory except for the initial clone.

## First-time setup on the cluster

Install uv on the Linux cluster login node first if it is not available (from any directory):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Follow the cluster guide's Miniconda setup. **Linux GPU cluster login node**, from any
directory, after checking storage usage. A login shell initializes the `module` command;
if it is missing after a normal SSH login, ask cluster staff to fix module initialization.
The `miniconda/24.1.2` module was available during the check.

```bash
module load miniconda
export CONDA_PKGS_DIRS="/scratch/$USER/.cache/conda/pkgs"
export UV_CACHE_DIR="/scratch/$USER/.cache/uv"
mkdir -p "$CONDA_PKGS_DIRS" "$UV_CACHE_DIR" "/scratch/$USER/envs"
conda create --override-channels --channel conda-forge \
  --prefix "/scratch/$USER/envs/laai-k1-python312" python=3.12
```

Reuse that interpreter on later setups; do not recreate an existing environment. Miniconda
provides Python 3.12 and uv installs this project's locked dependencies into `.venv`. The
Conda interpreter and project environment both stay on shared storage, so the job can use
them without depending on an interactive shell's activation.

For a **new checkout**, start in your own scratch directory:

```bash
cd "/scratch/$USER"
git clone https://github.com/jouchuki/laai-booster-k1-proj.git
cd laai-booster-k1-proj
```

For an **existing checkout**, enter its repository root instead. Do not move or duplicate
another person's checkout. From the repository root on the **Linux GPU cluster login node**:

```bash
bash scripts/setup_booster.sh assets # only the robot model is needed for training
uv sync --locked --group train --python "/scratch/$USER/envs/laai-k1-python312/bin/python"
uv run --no-sync --group train wandb login
```

Dependency installation runs on the login node; training runs in a SLURM job. Each launcher
uses `uv run --no-sync --group train` so it uses the prepared environment without downloading
packages on compute nodes. W&B login is optional; without a key training logs offline.
Keep API keys in your account's credential store or environment, never in Git or a batch file.
You can instead put `WANDB_API_KEY` in the Hex checkout's gitignored `.env`; the batch
launcher loads it with uv when the file exists. A local `.env` is not included in a Git
clone, so it must also be present in your private Hex checkout to use this method there.
The file is read as data and its contents are not printed by the launcher.

The `train` environment uses NumPy 2 and cannot be combined with `--group robot`, whose
Booster SDK requires NumPy below 2. uv enforces that separation; do not use `--all-groups`.

## Interactive compute

For debugging or running the direct train/export commands in `docs/commands.md`, request
an interactive allocation. **Linux GPU cluster login node**, from the repository root:

```bash
srun --partition=batch-gpu --gres=gpu:rtx4500:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=5 --mem=32G --time=00:30:00 --pty bash
```

Wait for the compute-node shell before running GPU commands, simulations or tests. Set
numerical-library thread counts within that allocation:

```bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
```

Type `exit` when finished to release the allocation. For CPU-only work, use a student CPU
partition: `short` (2 hours), `batch` (the guide says 4 days) or `long` (28 days). Check
`sinfo` for current limits; the live `batch` limit differed from the guide. Do not use a CPU
partition to bypass the student GPU policy.

## Free-bike comparison

The separate `LAAI-K1-Sit-Puky-Free` experiment compares a pretrained actor with scratch
training on an unsupported bike. Use the paired submission helper in
[docs/commands.md](../docs/commands.md#compare-free-bike-training-on-hex) from the login node.
It submits two jobs through `train_cpu.slurm`, each with 4 CPUs, 4 GB RAM and no GPU, with
an explicit 12-hour limit within the student `batch` partition's limit. Both use 256
environments, seed 42, 1,500 new iterations, W&B and the same final evaluation. The longer
limit accommodates free-body physics; it does not change the launcher's 8-hour default.

Logs and frozen comparison inputs live under `logs/laai_sit_free/`, separately from the
original `logs/laai_sit/` run. Source and run notes are in
[experiments/sit_free/](../experiments/sit_free/README.md). Check personal storage usage
before submission. Simulation and evaluation run inside each allocation, never on the
login node. The helper's `--test-only` validates scheduler requests without submitting.

## Submit CPU sit training

**Linux GPU cluster (Hex, CPU partition)**, from the repository root on the login node,
after completing the setup above. First submit three PPO iterations as a smoke test:

```bash
mkdir -p logs
sbatch --job-name=sit-cpu-smoke --time=00:15:00 cluster/train_cpu.slurm LAAI-K1-Sit-Puky \
  --env.scene.num-envs 8 --agent.max-iterations 3 --agent.run-name cpu-smoke \
  --agent.logger wandb
```

This starts an actual CPU job; it does not request or use a GPU. The launcher sets
`--gpu-ids None`, caps numerical-library threads to the allocation and loads an optional
ignored `.env` using uv. It does not run `nvidia-smi` or configure GPU rendering.
The smoke job uses W&B; put `WANDB_API_KEY` in this checkout's private `.env` for online
tracking. Without credentials the run is recorded offline for later synchronization.

After the smoke job finishes successfully, stage 0 uses 256 environments and the task's
default 1,500 iterations:

```bash
sbatch --job-name=sit-cpu-stage0 cluster/train_cpu.slurm LAAI-K1-Sit-Puky \
  --agent.run-name cpu-stage0
```

Stage 1 accepts the same CPU launcher and a stage 0 checkpoint:

```bash
sbatch --job-name=sit-cpu-stage1 cluster/train_cpu.slurm LAAI-K1-Sit-Puky-Lean \
  --init-from logs/laai_sit/YOUR_STAGE0_RUN/model_1499.pt --agent.run-name cpu-stage1
```

The default CPU request is 4 CPUs, 4 GB RAM and 8 hours in `batch`. CPU-only work is also
allowed in `short` (maximum 2 hours) or `long` (maximum 28 days); set the partition and time
together when overriding defaults. Use `batch` for a first training run because its duration
on Hex has not been measured. GPU options after the task ID are rejected by this launcher.

On the Linux laptop (Ryzen 7 260, five-thread budget), three-iteration CPU benchmarks used
1.51 GiB peak RAM and 2.60 seconds per iteration at 64 environments, and 1.58 GiB and
7.96 seconds at 256 environments. Extrapolating gives about 65 minutes or 3 hours 20 minutes
for 1,500 iterations on that laptop, not a measured Hex duration or evidence of convergence.
Fewer environments also mean fewer samples per PPO update than the GPU default of 4,096.

The first Hex smoke job (2026-11-17, job 54626) completed all three iterations with
`device=cpu`, exit code 0 and a SLURM-reported MaxRSS of 2.17 GiB. Total elapsed time was
10 minutes 13 seconds, including slow initial imports from shared storage, first-time Warp
CPU compilation and W&B synchronization. The iterations took 5.87, 2.19 and 2.87 seconds.
An initially quiet log can therefore reflect startup I/O before training begins; inspect
job state and output before interpreting it as stalled simulation. Full CPU training on
Hex has not yet been benchmarked.

To check scheduler acceptance without queueing a job:

```bash
sbatch --test-only cluster/train_cpu.slurm LAAI-K1-Sit-Puky
```

## Submit GPU sit training

**Linux GPU cluster**, from the repository root on the login node. First submit a short
GPU smoke run (32 environments, 3 iterations):

```bash
mkdir -p logs
sbatch --job-name=sit-smoke --time=00:30:00 cluster/train.slurm LAAI-K1-Sit-Puky \
  --env.scene.num-envs 32 --agent.max-iterations 3 --agent.run-name smoke \
  --agent.logger tensorboard
```

`logs/` must exist before submission because SLURM opens the output file before running the
script. Check `logs/slurm-<job-id>.out` for CUDA errors, GPU memory errors and successful
completion of all three iterations. Only after that succeeds, submit stage 0:

```bash
sbatch --job-name=sit-stage0 cluster/train.slurm LAAI-K1-Sit-Puky \
  --agent.run-name stage0
```

Stage 1 starts from a completed stage 0 checkpoint. Replace the example path with the actual
file in `logs/laai_sit/` and submit after stage 0 finishes:

```bash
sbatch --job-name=sit-stage1 cluster/train.slurm LAAI-K1-Sit-Puky-Lean \
  --init-from logs/laai_sit/YOUR_STAGE0_RUN/model_1499.pt --agent.run-name stage1
```

The same script also supports walk training:

```bash
sbatch --job-name=walk cluster/train.slurm LAAI-K1-Walk-Flat \
  --agent.run-name walk-baseline-ep
```

Any registered task works as the first argument. Sit checkpoints and metrics land in
`logs/laai_sit/<date>_<run-name>/`; walk uses `logs/laai_walk/`. Extra training options after
the task ID are forwarded, including `--init-from` and overrides of the default 4096 environments.

## Monitor jobs

**Linux GPU cluster**, from the repository root:

```bash
squeue -u "$USER"
tail -f logs/slurm-JOB_ID.out
```

Replace `JOB_ID` with the number returned by `sbatch`. Exit `tail` with Ctrl+C; the job keeps
running. To inspect a finished job or cancel a job, substitute its ID:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
scancel JOB_ID
```

## Tracking

Runs log to Weights & Biases (team `K1Booster`, set as `WANDB_ENTITY` in both launchers;
project `laai-k1`, grouped by experiment) once `wandb login`
has stored an API key on the cluster. Without a key, or with `WANDB_MODE=offline` set in
`train.slurm` because compute nodes have no internet, runs are logged to `logs/wandb/`. Upload
them afterwards from the Linux cluster login node:

```bash
uv run --no-sync --group train wandb sync logs/wandb/offline-run-*
```

Headless rendering (for rollout videos) needs `MUJOCO_GL=egl`, which `train.slurm` sets.
The headless job uses ordinary Python, not the macOS `mjpython` viewer launcher.

## Equipment bookings

The [booking guide](https://github.com/ci-group/slurm-cluster-doc/blob/main/booking_system.md)
applies separately to lab equipment, workstations and space: bookings are optional, give
priority and have a 40-hour outstanding total limit. Leave equipment clean and clear the
arena after your booking. Booking equipment does not allocate a SLURM GPU.

**Linux and macOS host**, from any directory, using your existing Hex SSH configuration:

```bash
ssh -L 8080:10.0.0.1:80 hex
```

While connected, open `http://localhost:8080/` and sign in with your cluster account. The
Optitrack system is an arena-booking accessory rather than a separate resource. Manufacturing
space is not currently bookable according to the guide.
