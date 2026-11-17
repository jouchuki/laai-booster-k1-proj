"""Summarize a finished training run for its notes and the experiment's run table.

Reads the TensorBoard files in a run folder (written with either logger), so it works offline
and gives every run the same numbers: each metric is the mean over the last N logged
iterations (default 50), which smooths out iteration-to-iteration noise.

Linux and macOS (from the repository root):
    uv run --group train python scripts/run_summary.py logs/laai_walk/2026-11-18_10-00-00_run3
    uv run --group train python scripts/run_summary.py logs/laai_walk/<run> --last 100

Prints:
  1. A "Simulation results" block for runs/runN/notes.md: headline numbers, every reward
     term (Episode_Reward/*), termination causes and task metrics (tracking error etc.).
  2. A row for the "Runs" table in experiments/<experiment>/README.md.
  3. The W&B run (link, or the offline run folder to `wandb sync`), if the run used W&B.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k1bike.paths import LOGS_DIR

REWARD = "Train/mean_reward"
EPISODE_LENGTH = "Train/mean_episode_length"
TASK_METRIC_PREFIXES = ("Metrics/", "Episode_Metrics/")


def read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    """All scalars of a run: tag -> [(iteration, value, wall time)], sorted by iteration."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not list(run_dir.glob("events.out.tfevents.*")):
        sys.exit(f"No TensorBoard files in {run_dir}")
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        # A resumed run can log an iteration twice; keep the last value per iteration.
        by_step = {e.step: (e.step, e.value, e.wall_time) for e in acc.Scalars(tag)}
        out[tag] = [by_step[s] for s in sorted(by_step)]
    return out


def tail_mean(series: list[tuple[int, float, float]], last: int) -> float:
    values = [v for _, v, _ in series[-last:]]
    return sum(values) / len(values)


def summarize(scalars: dict[str, list[tuple[int, float, float]]], last: int) -> dict:
    """Headline numbers and grouped metric means over the last `last` iterations."""
    if REWARD not in scalars:
        sys.exit(f"{REWARD} was never logged: did the run finish an iteration?")
    reward = scalars[REWARD]
    walls = [w for series in scalars.values() for _, _, w in series]

    def group(prefix: str) -> dict[str, float]:
        return {
            tag[len(prefix) :]: tail_mean(series, last)
            for tag, series in sorted(scalars.items())
            if tag.startswith(prefix)
        }

    return {
        "iterations": reward[-1][0] + 1,
        "wall_minutes": (max(walls) - min(walls)) / 60,
        "mean_reward": tail_mean(reward, last),
        "peak_reward": max(v for _, v, _ in reward),
        "episode_length": tail_mean(scalars[EPISODE_LENGTH], last)
        if EPISODE_LENGTH in scalars
        else None,
        "rewards": group("Episode_Reward/"),
        "terminations": group("Episode_Termination/"),
        "task_metrics": {
            name: value for prefix in TASK_METRIC_PREFIXES for name, value in group(prefix).items()
        },
        "last": min(last, len(reward)),
    }


def find_wandb_run(run_dir: Path, wandb_root: Path) -> Path | None:
    """W&B's local folder for this training run.

    rsl_rl stores the run folder as `log_dir` in the W&B config, which W&B's debug log records
    for online and offline runs alike.
    """
    for log in sorted(wandb_root.glob("*run-*/logs/debug.log"), reverse=True):
        if f"/{run_dir.resolve().name}'" in log.read_text(errors="replace"):
            return log.parent.parent
    return None


def wandb_link(wandb_dir: Path, project: str | None) -> str:
    """Web link for an uploaded run, or how to upload an offline one."""
    run_id = wandb_dir.name.rsplit("-", 1)[-1]
    if wandb_dir.name.startswith("offline-run-"):
        path = shlex.quote(os.path.relpath(wandb_dir))
        return f"offline, not uploaded: `uv run --group train wandb sync {path}`"
    try:
        import wandb

        api = wandb.Api(timeout=10)
        return api.run(f"{api.default_entity}/{project or 'laai-k1'}/{run_id}").url
    except Exception:  # noqa: BLE001  (no network or not logged in: give the id instead)
        return f"run id `{run_id}` in project `{project or 'laai-k1'}`"


def wandb_project(run_dir: Path) -> str | None:
    agent = run_dir / "params" / "agent.yaml"
    if agent.is_file():
        for line in agent.read_text().splitlines():
            if line.startswith("wandb_project:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    return None


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def notes_block(s: dict, run_dir: Path, wandb: str) -> str:
    lines = [
        f"Summary of `{run_dir}` (`scripts/run_summary.py`, mean of the last {s['last']} iterations)",
        "",
        f"- Iterations: {s['iterations']}, about {s['wall_minutes']:.0f} min",
        f"- Mean reward: {fmt(s['mean_reward'])} (peak {fmt(s['peak_reward'])})",
        f"- Episode length: {fmt(s['episode_length'], 4)} steps",
    ]
    lines += [f"- {name}: {fmt(value)}" for name, value in s["task_metrics"].items()]
    lines.append(f"- W&B: {wandb}")
    if s["rewards"]:
        lines += ["", "| Reward term | Per second of episode |", "|---|---|"]
        ranked = sorted(s["rewards"].items(), key=lambda kv: -abs(kv[1]))
        lines += [f"| {name} | {fmt(value)} |" for name, value in ranked]
    if s["terminations"]:
        lines += ["", "| Termination | Episodes per reset batch |", "|---|---|"]
        lines += [f"| {name} | {fmt(value)} |" for name, value in s["terminations"].items()]
    return "\n".join(lines)


def table_row(s: dict, run_dir: Path, wandb: str) -> str:
    try:
        date = datetime.strptime(run_dir.name[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        date = ""
    link = wandb if wandb.startswith("http") else "offline" if "offline" in wandb else "-"
    if link.startswith("http"):
        link = f"[run]({link})"
    return (
        f"| [N](runs/runN/notes.md) | {date} | <main change> | {s['iterations']} "
        f"| {fmt(s['mean_reward'])} | {fmt(s['episode_length'], 4)} "
        f"| <task metric> | <Booster MuJoCo> | <robot> | {link} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="logs/<experiment>/<date>_<run-name>/")
    parser.add_argument("--last", type=int, default=50, help="iterations to average")
    args = parser.parse_args()

    s = summarize(read_scalars(args.run_dir), args.last)
    found = find_wandb_run(args.run_dir, LOGS_DIR / "wandb")
    wandb = wandb_link(found, wandb_project(args.run_dir)) if found else "not logged to W&B"
    print("## Simulation results (for runs/runN/notes.md)\n")
    print(notes_block(s, args.run_dir, wandb))
    print("\n## Row for the Runs table in experiments/<experiment>/README.md\n")
    print(table_row(s, args.run_dir, wandb))


if __name__ == "__main__":
    main()
