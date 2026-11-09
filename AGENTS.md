# Agent instructions

## Linux and macOS commands

- Read `docs/commands.md` before adding or recommending setup, simulation, training,
  or deployment commands. It is the canonical command reference.
- Label command examples by execution target: **Linux host**, **macOS host**,
  **Linux GPU cluster**, **robot (Linux)**, or **Linux and macOS** when identical.
  State the working directory; default to the repository root.
- Keep differing Linux and macOS commands in separate fenced `bash` blocks.
  Do not put a macOS launcher in an unlabeled, general-purpose example.
- Use `uv run python` on Linux. On macOS, use `uv run mjpython` for scripts
  opening MuJoCo's native passive viewer, including `scripts/view_model.py`,
  native playback through `scripts/play.py`, and Booster's MuJoCo deployment.
- Use ordinary `uv run python` on both platforms for headless training, CPU
  smoke runs, tests, and scripts that do not open the native viewer. A browser
  viewer such as mjlab's `--viewer viser` also uses ordinary Python.
- Include the required dependency group: `--group train` for mjlab commands,
  `--group sim` for Booster's pretrained simulation, and `--group robot` for
  Linux-only SDK commands. Keep the project's Python 3.12 requirement.
- Separate OS compatibility from hardware requirements. GPU training runs on a
  Linux NVIDIA machine; local CPU smoke runs and viewing are different workflows.
  `sbatch` examples belong to the Linux SLURM cluster, not local macOS setup.
- Label `apt`/`apt-get` commands as Ubuntu/Debian or an Ubuntu container. Label
  `chflags` and Apple Silicon/Homebrew paths as macOS-specific.
- Keep `GDK_BACKEND=x11` an optional Linux GUI workaround, and `MUJOCO_GL=egl`
  a Linux NVIDIA headless-rendering setting. Do not apply either globally.
- SSH commands may be launched from either host OS; commands executed after
  connecting run on the robot's Linux system. Make that boundary explicit.
- Shell wrappers supporting both platforms must select the viewer launcher using
  `uname -s` (`Linux`/`Darwin`), resolve paths relative to the script, and forward
  arguments. Keep README examples and script help consistent when changing them.
- Do not claim macOS or GPU validation when only Linux static checks were run.

## Booster checkout paths

- Identify this repository by its files and script location, not its directory
  name. `booster` is an optional legacy subfolder, not a required root name.
- Reuse top-level `booster_assets/` and `booster_deploy/` checkouts first, then
  the older `booster/<repo>/` paths. Shell scripts share the resolver in
  `scripts/booster_paths.sh`; new clones go at the project root.
- uv's editable dependency is `booster_assets/` relative to `pyproject.toml`.
  For nested-only checkouts, setup creates relative top-level compatibility links.
  Update the lockfile when changing that source path. Do not move, reset, or
  duplicate a teammate's checkouts to impose a layout.

## Code navigation

Prefer the configured CodeGraph tools for structural questions (definitions,
callers, callees, signatures, impact, and task context). Use native search for
literal text and reads of known files. If CodeGraph is uninitialized, ask before
building its index. Follow the conditional graphify instructions in `CLAUDE.md`
when a graphify index exists.
