"""Watch a trained policy in mjlab's viewer, with our K1 tasks registered.

Wraps mjlab's `play` command. Run from the repository root and replace CHECKPOINT.pt.

Linux host (native viewer):
    uv run --group train python scripts/play.py LAAI-K1-Walk-Flat --checkpoint-file CHECKPOINT.pt --viewer native

macOS host (native viewer):
    uv run --group train mjpython scripts/play.py LAAI-K1-Walk-Flat --checkpoint-file CHECKPOINT.pt --viewer native

Linux and macOS (headless help; ordinary Python on both platforms):
    uv run --group train python scripts/play.py LAAI-K1-Walk-Flat --help

See docs/commands.md for CPU playback and the browser viewer alternative.
"""

import sys
from pathlib import Path

from mjlab.scripts.play import main

# Repository root on the import path, so the tasks in phases/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phases  # noqa: F401  (registers every phase's task with mjlab)

if __name__ == "__main__":
    main()
