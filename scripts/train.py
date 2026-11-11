"""Train a policy with mjlab, with our K1 tasks registered.

This wraps mjlab's own `train` command, so all of its options work.
Linux NVIDIA host or allocated GPU node (from the repository root):
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --env.scene.num-envs 4096
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --agent.run-name walk-baseline-ep

Linux and macOS (headless help; ordinary Python on both platforms):
    uv run --group train python scripts/train.py LAAI-K1-Walk-Flat --help

See docs/commands.md for CPU smoke runs.

Needs Linux + an NVIDIA GPU for real training (`uv sync --group train`). Without a GPU,
mjlab falls back to CPU, which is only useful for a quick smoke test with a few envs.
Checkpoints and logs go to logs/rsl_rl/<experiment>/<date>_<run-name>/ (gitignored).
Multi-GPU runs are not supported yet: our tasks are registered in this process only.
"""

import sys
from pathlib import Path

from mjlab.scripts.train import main

# Repository root on the import path, so the tasks in phases/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phases  # noqa: F401  (registers every phase's task with mjlab)

if __name__ == "__main__":
    main()
