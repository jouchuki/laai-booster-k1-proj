"""Train a policy with mjlab, with our K1 tasks registered.

This wraps mjlab's own `train` command, so all of its options work. Examples:
    uv run python scripts/train.py K1Bike-Walk-Flat --env.scene.num-envs 4096
    uv run python scripts/train.py K1Bike-Walk-Flat --agent.run-name walk-baseline-ep
    uv run python scripts/train.py K1Bike-Walk-Flat --help

Needs Linux + an NVIDIA GPU for real training (`uv sync --group train`). Without a GPU,
mjlab falls back to CPU, which is only useful for a quick smoke test with a few envs.
Checkpoints and logs go to logs/rsl_rl/<experiment>/<date>_<run-name>/ (gitignored).
Multi-GPU runs are not supported yet: our tasks are registered in this process only.
"""

from mjlab.scripts.train import main

import k1bike.tasks  # noqa: F401  (registers the K1 tasks with mjlab)

if __name__ == "__main__":
    main()
