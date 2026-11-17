"""Train a policy with mjlab, with our K1 tasks registered.

This wraps mjlab's own `train` command, so all of its options work.
Linux NVIDIA host or allocated GPU node (from the repository root):
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --env.scene.num-envs 4096
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --agent.run-name walk-baseline-ep

On Hex, submit cluster/train.slurm (GPU) or cluster/train_cpu.slurm (CPU) from the login
node, or use an interactive SLURM allocation. See cluster/README.md for resource limits.

Start from an earlier checkpoint instead of from scratch (the next curriculum stage, for
example sit stage 1 from a stage 0 policy). Both tasks must log into the same experiment
folder and have the same inputs and outputs; training continues from that checkpoint's
weights, optimizer state and iteration count, in a new run folder:
    uv run --group train python scripts/train.py LAAI-K1-Sit-Puky-Lean \\
        --init-from logs/laai_sit/2026-11-16_10-00-00_run1/model_1499.pt --env.scene.num-envs 4096

Linux and macOS (headless help; ordinary Python on both platforms):
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --help

Experiment tracking: our tasks log to Weights & Biases (project `laai-k1`), one group per
experiment (`laai_walk`, `laai_sit`), tagged with the task id. Every metric is also written as
TensorBoard files in the run folder. Not logged in (`wandb login` or WANDB_API_KEY)? The run
is logged offline under logs/wandb/ and can be uploaded later with `wandb sync`. Opt out with
`--agent.logger tensorboard`. Summarize a finished run for its notes with
scripts/run_summary.py.
For a key stored in the repository root's ignored .env, add `--env-file .env` to `uv run`;
Both cluster launchers do this automatically when that file exists in the cluster checkout.

See docs/commands.md for CPU smoke runs.

GPU training needs Linux + NVIDIA (`uv sync --group train`). CPU training also works with
--gpu-ids None; use fewer environments and measure throughput before choosing a run budget.
Checkpoints and logs go to logs/<experiment>/<date>_<run-name>/ (gitignored).
Multi-GPU runs are not supported yet: our tasks are registered in this process only.
"""

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from mjlab.scripts.train import main

# Repository root on the import path, so the tasks in experiments/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments  # noqa: F401  (registers every experiment's task with mjlab)
from k1bike.paths import LOGS_DIR


def init_from_args(argv: list[str]) -> list[str]:
    """Turn `--init-from <checkpoint>` into mjlab's resume options.

    mjlab resumes from logs/<experiment>/<run>/<checkpoint> of the task being trained, so the
    checkpoint must sit in that task's experiment folder.
    """
    if "--init-from" not in argv:
        return argv
    i = argv.index("--init-from")
    if i + 1 >= len(argv):
        sys.exit("--init-from needs a checkpoint file, e.g. logs/laai_sit/<run>/model_1499.pt")
    checkpoint = Path(argv[i + 1]).resolve()
    rest = argv[:i] + argv[i + 2 :]
    if not checkpoint.is_file():
        sys.exit(f"--init-from: no such checkpoint: {checkpoint}")
    from mjlab.tasks.registry import load_rl_cfg

    task = rest[1] if len(rest) > 1 and not rest[1].startswith("-") else None  # first argument
    experiment = load_rl_cfg(task).experiment_name if task else None
    log_root = LOGS_DIR.resolve()
    if checkpoint.parent.parent != log_root / str(experiment):
        sys.exit(
            f"--init-from: {task} resumes only from {log_root / str(experiment)}/<run>/; "
            f"{checkpoint} is elsewhere (copy its run folder there)."
        )
    return rest + [
        "--agent.resume",
        "True",
        "--agent.load-run",
        f"^{re.escape(checkpoint.parent.name)}$",
        "--agent.load-checkpoint",
        f"^{re.escape(checkpoint.name)}$",
    ]


def option_value(argv: list[str], name: str) -> str | None:
    """Value of a command-line option given as `name value` or `name=value`, else None."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def wandb_logged_in() -> bool:
    """True if W&B has an API key (WANDB_API_KEY or a `wandb login` entry in ~/.netrc)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import wandb
        from wandb.sdk.lib import wbauth

        return wbauth.read_netrc_auth(host=wandb.Settings().base_url) is not None
    except Exception:  # noqa: BLE001  (unknown wandb version: treat as not logged in)
        return False


def wandb_env(
    argv: list[str], environ: Mapping[str, str], logged_in: bool, default_logger: str | None
) -> dict[str, str]:
    """Environment variables that organize a W&B run; empty if this run does not use W&B.

    Groups the run by experiment (the log folder name), tags it with the task id, keeps W&B's
    local files in logs/wandb/, and logs offline when there is no API key, so a job never
    stops at a login prompt. Variables already set in `environ` win.
    """
    if "-h" in argv or "--help" in argv:
        return {}
    task = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
    if (option_value(argv, "--agent.logger") or default_logger) != "wandb":
        return {}
    env: dict[str, str] = {"WANDB_DIR": str(LOGS_DIR)}
    experiment = option_value(argv, "--agent.experiment-name")
    if experiment is None and task is not None:
        from mjlab.tasks.registry import load_rl_cfg

        experiment = load_rl_cfg(task).experiment_name
    if experiment:
        env["WANDB_RUN_GROUP"] = experiment
    if task:
        tags = {t for t in environ.get("WANDB_TAGS", "").split(",") if t} | {task}
        env["WANDB_TAGS"] = ",".join(sorted(tags))
    if not logged_in and "WANDB_MODE" not in environ:
        env["WANDB_MODE"] = "offline"
    return {k: v for k, v in env.items() if k == "WANDB_TAGS" or k not in environ}


def patch_wandb_settings() -> None:
    """Let rsl_rl start W&B with current wandb versions.

    rsl_rl's WandbLogWriter calls `wandb.Settings(start_method="thread")`, a field newer wandb
    releases (0.31 here) removed, so `wandb.init` fails before training starts. Drop that field;
    remove this once rsl_rl stops passing it.
    """
    import wandb

    if "start_method" in wandb.Settings.model_fields:
        return
    original = wandb.Settings

    class Settings(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs):
            kwargs.pop("start_method", None)
            super().__init__(**kwargs)

    wandb.Settings = Settings


def default_logger(argv: list[str]) -> str | None:
    """The logger the task's config chooses (before command-line overrides)."""
    if len(argv) < 2 or argv[1].startswith("-"):
        return None
    from mjlab.tasks.registry import load_rl_cfg

    return load_rl_cfg(argv[1]).logger


if __name__ == "__main__":
    # Logs in logs/<experiment>/ instead of mjlab's default logs/rsl_rl/<experiment>/.
    if not any(arg.startswith("--log-root") for arg in sys.argv[1:]):
        sys.argv += ["--log-root", str(LOGS_DIR)]
    sys.argv = init_from_args(sys.argv)
    help_wanted = "-h" in sys.argv or "--help" in sys.argv
    tracking = wandb_env(
        sys.argv,
        os.environ,
        logged_in=wandb_logged_in(),
        default_logger=None if help_wanted else default_logger(sys.argv),
    )
    if tracking:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.update(tracking)
        patch_wandb_settings()
        if tracking.get("WANDB_MODE") == "offline":
            print(
                "[INFO] W&B: no API key found, logging offline to logs/wandb/. Upload later with\n"
                "       uv run --group train wandb sync logs/wandb/offline-run-*"
            )
    main()
