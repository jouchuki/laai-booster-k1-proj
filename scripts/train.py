"""Train a policy with mjlab, with our K1 tasks registered.

This wraps mjlab's own `train` command, so all of its options work.
Linux NVIDIA host or allocated GPU node (from the repository root):
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --env.scene.num-envs 4096
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --agent.run-name walk-baseline-ep

Start from an earlier checkpoint instead of from scratch (the next curriculum stage, for
example sit stage 1 from a stage 0 policy). Both tasks must log into the same experiment
folder and have the same inputs and outputs; training continues from that checkpoint's
weights, optimizer state and iteration count, in a new run folder:
    uv run --group train python scripts/train.py LAAI-K1-Sit-Puky-Lean \\
        --init-from logs/laai_sit/2026-11-16_10-00-00_run1/model_1499.pt --env.scene.num-envs 4096

Linux and macOS (headless help; ordinary Python on both platforms):
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --help

See docs/commands.md for CPU smoke runs.

Needs Linux + an NVIDIA GPU for real training (`uv sync --group train`). Without a GPU,
mjlab falls back to CPU, which is only useful for a quick smoke test with a few envs.
Checkpoints and logs go to logs/<experiment>/<date>_<run-name>/ (gitignored).
Multi-GPU runs are not supported yet: our tasks are registered in this process only.
"""

import re
import sys
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


if __name__ == "__main__":
    # Logs in logs/<experiment>/ instead of mjlab's default logs/rsl_rl/<experiment>/.
    if not any(arg.startswith("--log-root") for arg in sys.argv[1:]):
        sys.argv += ["--log-root", str(LOGS_DIR)]
    sys.argv = init_from_args(sys.argv)
    main()
