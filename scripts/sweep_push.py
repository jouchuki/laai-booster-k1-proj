"""Plan, submit and rank a push reward sweep (experiments/push/sweep.py explains the stages).

Linux GPU cluster, repository root on the Hex login node (submits one job per run):
    uv run --no-sync --group train python scripts/sweep_push.py plan groups
    uv run --no-sync --group train python scripts/sweep_push.py submit groups --seeds 2 --test-only
    uv run --no-sync --group train python scripts/sweep_push.py submit groups --seeds 2
    uv run --no-sync --group train python scripts/sweep_push.py submit terms --base simple+gait+motors
    uv run --no-sync --group train python scripts/sweep_push.py collect logs/laai_push/sweeps/<sweep>

Linux and macOS, repository root (each run trained here, one after another; for smoke tests):
    uv run --group train python scripts/sweep_push.py submit groups --local --only simple,all --iterations 2

Every run is LAAI-K1-Push-Puky-Form with reward weights overridden on the command line.
The sweep's manifest (runs, commands, job ids, git revision) and its ranked results go to
logs/laai_push/sweeps/<sweep>/; each run logs as usual to logs/laai_push/<date>_<run-name>/,
and to W&B group <sweep>.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.push import sweep
from k1bike.paths import LOGS_DIR

EXPERIMENT_DIR = LOGS_DIR / "laai_push"
SWEEPS_DIR = EXPERIMENT_DIR / "sweeps"


def form_defaults() -> dict[str, float]:
    from experiments.push import task as push

    rewards = push.push_env_cfg(shaped=True, form=True).rewards
    return {name: term.weight for name, term in rewards.items()}


def train_args(config, defaults, *, seed, iterations, run_name, init_policy_from=None):
    from experiments.push.task import FORM_TASK_ID

    args = [FORM_TASK_ID, *sweep.overrides(config, defaults)]
    args += ["--agent.max-iterations", str(iterations), "--agent.seed", str(seed)]
    args += ["--agent.run-name", run_name]
    if init_policy_from:
        args += ["--agent.init-policy-from", str(init_policy_from)]
    return args


def plan(args) -> tuple[str, list[dict]]:
    configs = sweep.make_stage(args.stage, args.base)
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.name for c in configs}
        if unknown:
            sys.exit(f"--only: no such configs in this stage: {sorted(unknown)}")
        configs = [c for c in configs if c.name in wanted]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"sweep-{args.stage}-{stamp}"
    defaults = form_defaults()
    runs = []
    for config in configs:
        for seed in range(args.seed, args.seed + args.seeds):
            run_name = f"{name}-{config.name}-s{seed}"
            runs.append(
                {
                    "config": config.name,
                    "seed": seed,
                    "run_name": run_name,
                    "terms": list(config.terms),
                    "scales": {t: s for t, s in config.scales.items() if s != 0.0},
                    "train_args": train_args(
                        config,
                        defaults,
                        seed=seed,
                        iterations=args.iterations,
                        run_name=run_name,
                        init_policy_from=args.init_policy_from,
                    ),
                }
            )
    return name, runs


def job_command(args, sweep_name, run) -> list[str]:
    if args.local:
        return [
            sys.executable,
            "scripts/train.py",
            *run["train_args"],
            "--gpu-ids",
            "None",
            "--env.scene.num-envs",
            str(args.num_envs or 8),
            "--agent.logger",
            "tensorboard",
        ]
    script = "cluster/train_cpu.slurm" if args.cpu else "cluster/train.slurm"
    command = ["sbatch", "--parsable", f"--job-name=ps-{run['config']}-s{run['seed']}"]
    if args.time:
        command.append(f"--time={args.time}")
    if args.test_only:
        command.append("--test-only")
    # sbatch splits --export at commas, so W&B gets one tag here (plus the task id).
    command += [f"--export=ALL,WANDB_RUN_GROUP={sweep_name},WANDB_TAGS={args.stage}", script]
    command += run["train_args"]
    if args.num_envs:
        command += ["--env.scene.num-envs", str(args.num_envs)]
    return command


def cmd_plan(args) -> None:
    name, runs = plan(args)
    configs = {}
    for run in runs:
        configs.setdefault(run["config"], run)
    print(f"{name}: {len(configs)} configs x {args.seeds} seeds = {len(runs)} runs\n")
    for config, run in configs.items():
        scaled = {t: s for t, s in run["scales"].items() if s != 1.0}
        extra = f"  scaled {scaled}" if scaled else ""
        print(f"{config:28s} {len(run['terms']):2d} terms: {' '.join(run['terms'])}{extra}")


def cmd_submit(args) -> None:
    name, runs = plan(args)
    folder = SWEEPS_DIR / name
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, cwd=ROOT, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=False,
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    manifest = {
        "sweep": name,
        "stage": args.stage,
        "base": args.base if args.stage != "groups" else None,
        "task": "LAAI-K1-Push-Puky-Form",
        "iterations": args.iterations,
        "git_revision": revision,
        "uncommitted_changes": dirty,
        "gates": sweep.GATES,
        "mode": "local" if args.local else ("cpu" if args.cpu else "gpu"),
        "runs": runs,
    }
    if not args.test_only:
        folder.mkdir(parents=True, exist_ok=False)
    if dirty:
        print(
            "[WARN] Uncommitted changes: the manifest's git revision does not fully describe them."
        )
    for run in runs:
        command = job_command(args, name, run)
        run["command"] = command
        print(shlex.join(command), flush=True)
        if args.local:
            result = subprocess.run(command, check=False, cwd=ROOT)
            run["exit_code"] = result.returncode
        else:
            # No shell: option values stay separate arguments.
            result = subprocess.run(command, check=False, cwd=ROOT, text=True, capture_output=True)
            if result.returncode != 0:
                print(result.stderr.strip(), file=sys.stderr)
                if not args.test_only:
                    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                sys.exit(f"sbatch failed for {run['run_name']}; earlier jobs stay submitted")
            output = (result.stdout + result.stderr).strip()
            if output:
                print(output, flush=True)
            run["job"] = result.stdout.strip().split(";")[0] or None
        if not args.test_only:
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.test_only:
        print(f"\nManifest: {folder / 'manifest.json'}")
        print(f"Rank when done: python scripts/sweep_push.py collect {folder.relative_to(ROOT)}")


def cmd_collect(args) -> None:
    folder = Path(args.sweep).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = sweep.collect(manifest, EXPERIMENT_DIR)
    (folder / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    text = sweep.results_markdown(rows, manifest["sweep"])
    (folder / "results.md").write_text(text)
    print(text)
    print(f"Written: {folder / 'results.md'} and results.json")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "submit"):
        p = sub.add_parser(command, help=f"{command} a stage of the sweep")
        p.add_argument("stage", choices=sorted(sweep.STAGES))
        p.add_argument(
            "--base",
            default="all",
            help='reward set for the terms and weights stages, e.g. "simple+gait+motors" '
            'or "all-form-arm_effort" (sets: core, task, simple, all, '
            + ", ".join(sweep.GROUPS)
            + "; or term names)",
        )
        p.add_argument("--only", help="comma-separated config names to keep")
        p.add_argument("--seeds", type=int, default=2, help="seeds per config (default 2)")
        p.add_argument("--seed", type=int, default=42, help="first seed (default 42)")
        p.add_argument("--iterations", type=int, default=3000)
        p.add_argument("--init-policy-from", type=Path, help="actor checkpoint for every run")
        if command == "submit":
            where = p.add_mutually_exclusive_group()
            where.add_argument("--local", action="store_true", help="train here, sequentially")
            where.add_argument("--cpu", action="store_true", help="cluster/train_cpu.slurm")
            p.add_argument("--num-envs", type=int, help="override the job's environment count")
            p.add_argument("--time", help="SLURM time limit, e.g. 08:00:00")
            p.add_argument("--test-only", action="store_true", help="sbatch --test-only")
    c = sub.add_parser("collect", help="rank a sweep's finished runs")
    c.add_argument("sweep", help="logs/laai_push/sweeps/<sweep>")
    args = parser.parse_args(argv)
    if getattr(args, "local", False) and args.test_only:
        parser.error("--test-only is for SLURM submissions")
    {"plan": cmd_plan, "submit": cmd_submit, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
