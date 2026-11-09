"""Watch a trained policy in mjlab's viewer, with our K1 tasks registered.

Wraps mjlab's `play` command. Run from the repository root and replace CHECKPOINT.pt.

Linux host (native viewer):
    uv run --group train python scripts/play.py K1Bike-Walk-Flat --checkpoint-file CHECKPOINT.pt --viewer native

macOS host (native viewer):
    uv run --group train mjpython scripts/play.py K1Bike-Walk-Flat --checkpoint-file CHECKPOINT.pt --viewer native

Linux and macOS (headless help; ordinary Python on both platforms):
    uv run --group train python scripts/play.py K1Bike-Walk-Flat --help

See docs/commands.md for CPU playback and the browser viewer alternative.
"""

from mjlab.scripts.play import main

import k1bike.tasks  # noqa: F401  (registers the K1 tasks with mjlab)

if __name__ == "__main__":
    main()
