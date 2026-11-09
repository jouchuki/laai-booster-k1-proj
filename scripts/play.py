"""Watch a trained policy in mjlab's viewer, with our K1 tasks registered.

Wraps mjlab's `play` command. Examples:
    uv run python scripts/play.py K1Bike-Walk-Flat --checkpoint-file logs/rsl_rl/k1_walk/<run>/model_1000.pt
    uv run python scripts/play.py K1Bike-Walk-Flat --help
"""

from mjlab.scripts.play import main

import k1bike.tasks  # noqa: F401  (registers the K1 tasks with mjlab)

if __name__ == "__main__":
    main()
