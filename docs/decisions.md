# Decision log

Newest first. One entry per decision: what, why, what else was considered.

## 2026-11-12: The K1 collides with itself and with the bike

**Decision:** Every task keeps the K1's self-collision as in Booster's model (174 geom
pairs) and lets the robot collide with every bike collision geom. The only exception is
each forearm against the fork, because the welded hand sits inside the handlebar's collision
capsule; it goes once Vlad's hand model is attached and the hands touch the bar through their
own geoms. Tests: `test_walk_task_keeps_self_collision`, `test_robot_collides_with_itself_and_the_bike`.

**Why:** The policy must learn the physics the real robot has. A robot whose legs pass
through each other, or through the frame, learns sitting and pushing motions the K1 cannot do.

**Considered:** Training without self-collision for speed (mjlab's own humanoid tasks do;
planned as walk run 3 and dropped). Collision geoms of the bike stay the measured primitives
(capsules, cylinders) of `assets/bike/sketchfab_mujoco/`, not its visual mesh; they need
updating when the bike model changes.

## 2026-11-11: One folder per project phase

**Decision:** Everything that belongs to one phase of the plan lives in `phases/<phase>/`:
the mjlab task (`task.py`), the export to the robot's format (`export.py`), the booster_deploy
task files (`deploy/`), one folder per training run (`runs/runN/`) and the phase report
(`README.md`). Shared tools stay outside: `scripts/` (setup, training, viewers), `deploy/`
(install, robot setup, dry run), `src/k1bike/` (paths and scenes), `docs/` (commands,
decisions). Names use the project prefix: mjlab task `LAAI-K1-<Phase>-...`, log folder and
booster_deploy task `laai_<phase>`.

**Why:** The project adds one skill per phase (README plan) and every skill goes through the
same steps: train, export, test in Booster's simulator, test on the robot. One folder per
phase keeps a skill's code, results and robot logs together, so a phase can be read, run and
handed over on its own.

**Considered:** Grouping by step (all tasks together, all runs together, all reports
together). That spread one phase over five places.

## 2026-11-11: Walk policy uses booster_deploy's `k1_walk` interface

**Decision:** The phase 1 walk task (`phases/walk/task.py`) trains with the observation,
action, PD gains, default pose and joint order of Booster's `k1_walk` deploy task: 690 inputs
(69 per frame x 10 frames), 20 outputs, action scale 0.25. `phases/walk/export.py` writes a
TorchScript policy that reorders mjlab's layout into booster_deploy's and checks the result
against mjlab's own actions.

**Why:** booster_deploy's existing `LocomotionPolicy` then runs our policy unchanged, in MuJoCo
and on the robot, without forking booster_deploy.

**Result:** Run 2 walks on the robot (see `phases/walk/README.md`). Robot-side additions live in
`phases/walk/deploy/laai_walk.py`: speed limits, safety watchdog, smooth handover, ankle damping variant.

**Considered:** mjlab's default observation layout plus a custom policy class in a
booster_deploy fork.

## 2026-11-09: Train with mjlab; runs 1-2 on Google Colab

**Decision:** Train with [mjlab](https://github.com/mujocolab/mjlab). Walk runs 1 and 2 ran
on a Colab GPU (their notebooks are in `phases/walk/runs/`); new runs use `scripts/train.py`
on the Linux GPU cluster.

**Why:** mjlab trains in MuJoCo, the same physics engine as booster_deploy's simulator and our
bike model, runs thousands of environments in parallel on one GPU, and Colab was available
immediately.

## 2026-11-09: Reuse Booster checkouts independently of the project root name

**Decision:** Prefer Booster checkouts directly under this project's root and
fall back to the older `booster/` subfolder. Resolve paths from the script's
location rather than the current directory or the root directory's name.

**Why:** Existing teammates' clones use both layouts. uv must resolve the local
`booster-assets` dependency even when it is only preparing a development command
such as `pre-commit install`.

**Implementation:** uv uses the top-level `booster_assets` path. Setup reuses
existing checkouts, creates relative compatibility links for nested-only clones,
and puts new clones at the project root. The viewer wrapper shares the same path
resolver. Existing Git checkouts and their revisions are preserved.

**Considered:** Requiring every teammate to relocate or reclone the repositories.
Reusing the existing layout avoids that work and unnecessary downloads.

## 2026-11-09: Separate commands by platform and execution target

**Decision:** Keep a canonical command reference in `docs/commands.md`, with
explicit Linux host, macOS host, Linux GPU cluster, and robot sections. Put the
same conventions in root `AGENTS.md`, referenced by `CLAUDE.md`.

**Why:** Native MuJoCo viewers require `mjpython` on macOS and ordinary Python
on Linux. Headless training and shared development commands use ordinary Python
on both platforms. Mixed examples were sending teammates through incompatible
launchers and conflating host, cluster, container, and robot environments.

**Implementation:** Separate viewer examples in docs and script help. The
pretrained-task wrapper selects the launcher using `uname -s` and resolves the
Booster checkout relative to the repository root. Keep Linux display workarounds
opt-in and keep legacy ROS/Docker experiments outside the main training workflow.

**Considered:** Duplicating every script into Linux and macOS versions. Shared
implementations with explicit examples and a small OS-dispatching wrapper avoid
maintaining two copies of the same behavior.
