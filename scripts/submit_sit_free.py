"""Submit a matched pretrained/scratch CPU comparison from the Hex login node.

Linux GPU cluster, repository root:
    uv run --no-sync --group train python scripts/submit_sit_free.py CHECKPOINT.pt --test-only
    uv run --no-sync --group train python scripts/submit_sit_free.py CHECKPOINT.pt
See docs/commands.md. This script prepares files and submits jobs; it runs no simulation.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experiments/__init__.py",
    "experiments/sit/task.py",
    "experiments/sit_free/__init__.py",
    "experiments/sit_free/task.py",
    "experiments/sit_free/runner.py",
    "experiments/sit_free/evaluate.py",
    "experiments/walk/task.py",
    "scripts/train.py",
    "scripts/submit_sit_free.py",
    "cluster/train_cpu.slurm",
    "pyproject.toml",
    "uv.lock",
)


def submission_commands(checkpoint, group, *, test_only=False):
    common = [
        "LAAI-K1-Sit-Puky-Free",
        "--env.scene.num-envs",
        "256",
        "--agent.max-iterations",
        "1500",
        "--agent.seed",
        "42",
        "--agent.logger",
        "wandb",
    ]
    commands = {}
    for mode in ("pretrained", "scratch"):
        command = ["sbatch", "--parsable", "--time=12:00:00", f"--job-name=sit-free-{mode}"]
        if test_only:
            command.append("--test-only")
        command += [
            f"--export=ALL,WANDB_RUN_GROUP={group},WANDB_TAGS={mode}",
            "cluster/train_cpu.slurm",
            *common,
            "--agent.run-name",
            f"{group}-{mode}",
        ]
        if mode == "pretrained":
            command += ["--agent.init-policy-from", str(checkpoint)]
        commands[mode] = command
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--test-only", action="store_true", help="SLURM validation; submit no jobs")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        parser.error("checkpoint must be a file")
    group = "free-bike-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    comparison = ROOT / "logs/laai_sit_free/comparisons" / group
    frozen = comparison / checkpoint.name
    if args.test_only:
        commands = submission_commands(checkpoint, group, test_only=True)
        manifest = None
    else:
        comparison.mkdir(parents=True, exist_ok=False)
        shutil.copy2(checkpoint, frozen)
        commands = submission_commands(frozen, group)
        with tarfile.open(comparison / "source.tar.gz", "w:gz") as archive:
            for name in SOURCE_FILES:
                archive.add(ROOT / name, arcname=name)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        manifest = {
            "group": group,
            "experiment": "laai_sit_free",
            "source_checkpoint": str(checkpoint),
            "frozen_checkpoint": str(frozen),
            "checkpoint_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "git_revision": revision,
            "source_snapshot": "source.tar.gz",
            "num_envs": 256,
            "iterations_per_run": 1500,
            "seed": 42,
            "commands": commands,
            "jobs": {},
        }
    for mode, command in commands.items():
        print(shlex.join(command), flush=True)
        if manifest:
            (comparison / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # No shell interpolation; filenames and option values remain separate arguments.
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        print(result.stdout.strip(), flush=True)
        if result.stderr:
            print(result.stderr.strip(), flush=True)
        if manifest:
            manifest["jobs"][mode] = result.stdout.strip().split(";")[0]
            (comparison / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest:
        print(f"Comparison manifest: {comparison / 'manifest.json'}")


if __name__ == "__main__":
    main()
