"""Watch a trained policy in mjlab's viewer, with our K1 tasks registered.

Wraps mjlab's `play` command. Run from the repository root and replace CHECKPOINT.pt.

Linux host (native viewer):
    uv run --group train python scripts/play.py LAAI-K1-Walk-Flat --checkpoint-file CHECKPOINT.pt --viewer native

macOS host (native viewer):
    uv run --group train mjpython scripts/play.py LAAI-K1-Walk-Flat --checkpoint-file CHECKPOINT.pt --viewer native

Linux and macOS (headless help; ordinary Python on both platforms):
    uv run --group train python scripts/play.py LAAI-K1-Walk-Flat --help

For tasks with a velocity command, Up/Down arrows in the native viewer set the forward speed
command (printed in the terminal); the browser viewer (--viewer viser) has sliders instead.

See docs/commands.md for CPU playback and the browser viewer alternative.
"""

import dataclasses
import sys
from pathlib import Path

from mjlab.scripts.play import main
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.viewer import NativeMujocoViewer

# mjlab's viser joystick builds each axis's "Max" slider on [0.1, 10] starting at the axis's
# upper range, so it crashes when a range is fixed at zero (the push task's sideways speed and
# turn rate). Widen such ranges to 0.1 only while the panel is built; sampling is unchanged.
_create_gui = UniformVelocityCommand.create_gui


def _create_gui_zero_safe(self, *args, **kwargs):
    ranges = self.cfg.ranges
    gui_ranges = dataclasses.replace(
        ranges,
        **{
            axis: (lo, max(hi, 0.1))
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
            for lo, hi in [getattr(ranges, axis)]
        },
    )
    self.cfg.ranges = gui_ranges
    try:
        return _create_gui(self, *args, **kwargs)
    finally:
        self.cfg.ranges = ranges


UniformVelocityCommand.create_gui = _create_gui_zero_safe

# Native viewer: Up/Down arrows set the forward speed command, which then holds instead of
# resampling (sideways speed and turn rate are held at zero). Before the first press the commands
# resample as in training. The steps are 0 plus 0.05 m/s steps across the task's lin_vel_x range,
# so the push task's 0.15-0.6 m/s range gives 0, 0.15, 0.2, ... 0.6.
_keyboard = {"speed": None}  # None: commands as in training
_velocity_terms: list = []  # every velocity command term, recorded on its first compute
_compute = UniformVelocityCommand.compute


def _compute_with_keyboard(self, dt, env_ids=None):
    if self not in _velocity_terms:
        _velocity_terms.append(self)
    _compute(self, dt, env_ids)
    speed = _keyboard["speed"]
    if speed is not None:
        self.vel_command_b[:, 0] = speed
        self.vel_command_b[:, 1:] = 0.0


UniformVelocityCommand.compute = _compute_with_keyboard


def _speed_steps() -> list[float]:
    lo, hi = _velocity_terms[0].cfg.ranges.lin_vel_x
    steps = {0.0} | {round(lo + 0.05 * i, 3) for i in range(round((hi - lo) / 0.05) + 1)}
    return sorted(steps)


def _speed_key(key: int) -> None:
    from mjlab.viewer.native.keys import KEY_DOWN, KEY_UP

    if key not in (KEY_UP, KEY_DOWN) or not _velocity_terms:
        return
    steps = _speed_steps()
    current = _keyboard["speed"]
    if current is None:  # first press: start from standing still
        index = steps.index(0.0) + (1 if key == KEY_UP else 0)
    else:
        index = min(range(len(steps)), key=lambda i: abs(steps[i] - current))
        index += 1 if key == KEY_UP else -1
    _keyboard["speed"] = steps[max(0, min(index, len(steps) - 1))]
    print(f"[keyboard] forward speed command: {_keyboard['speed']:.2f} m/s", flush=True)


_native_init = NativeMujocoViewer.__init__


def _native_init_with_keys(self, *args, key_callback=None, **kwargs):
    def callback(key: int) -> None:
        _speed_key(key)
        if key_callback is not None:
            key_callback(key)

    _native_init(self, *args, key_callback=callback, **kwargs)


NativeMujocoViewer.__init__ = _native_init_with_keys

# Repository root on the import path, so the tasks in experiments/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments  # noqa: F401  (registers every experiment's task with mjlab)
from k1bike.paths import LOGS_DIR

if __name__ == "__main__":
    # Logs in logs/<experiment>/ instead of mjlab's default logs/rsl_rl/<experiment>/.
    if not any(arg.startswith("--log-root") for arg in sys.argv[1:]):
        sys.argv += ["--log-root", str(LOGS_DIR)]
    main()
