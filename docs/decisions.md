# Decision log

Newest first. One entry per decision: what, why, what else was considered.

## 2026-11-12: Saddle collision follows the visible saddle

**Decision:** Every scene loads a bike through `load_bike()` (`src/k1bike/scene.py`), which
replaces the saddle's flat collision box with 12 thin convex slices fitted by ray-casting the
visible saddle (`src/k1bike/saddle.py`). The box keeps its mass and no longer collides.
The seated poses were re-solved on the fitted saddles, and the pose solver now starts from
several trunk leans and keeps the best fit.

**Why:** The K1 should sit on the saddle as it looks and as the real one is shaped. On the
flat box, set at the saddle's highest point, the pelvis hovered 13-14 mm above the Puky's
dished saddle (4-5 mm on the yellow bike) while pressing on nothing visible. Now the visible
gap is under 1 mm and the K1 puts about 96 N (Puky) and 86 N (yellow bike) of its 193 N on
the saddle; tests check both.

**Follow-up the same day:** the K1 now sits in the saddle's dip (a `saddle_dip` site the
solver aims the pelvis at), the Puky's seat post is shortened below the dip (the K1 had been
sitting on it), the K1 + bike scenes have one floor instead of two (Booster's `ground` plane
doubled every foot and wheel contact), and the solved poses keep the feet 5 mm up so the
legs are relaxed. Holding the pose, the Puky's saddle carries 86% of the K1's weight, the
feet 10% and the hands 4% (yellow bike 67/23/9), like a rider on a balance bike.

**Considered:** Lowering the box to the dish (the rider would sink into the visible rim and
could slide off the flat top); one convex hull of the saddle (fills the dish, the same
problem as the box); fitting the slices in the bike converters (only for the two converted
bikes, and every bike would have to be regenerated).

## 2026-11-12: The real bike is the default; the K1 can sit on any bike

**Decision:** `models/puky_bike` (the calibrated scan of the bike we will use) is the default
bike for every scene, the viewer and the seated K1 (`DEFAULT_BIKE` in `src/k1bike/paths.py`).
Seated poses are stored per bike, `models/k1_on_bike/<bike>.json`, solved with
`scripts/solve_seated_pose.py --bike <bike>`; `k1_seated_bike_scene(bike_xml)` loads the one
for its bike, and the seat tests run on every bike that has one. The solver now also targets
the feet 6 cm in front of the saddle and a 20 deg forward trunk lean, and refuses to write a
pose whose fit fails.

**Why:** Training and robot tests must use the bike the robot will ride, and the next bike
(or a re-measured one) should only need a solved pose, not code changes. On the real bike the
saddle is 7.6 cm in front of the rear axle: an upright K1 with its feet behind the hips
pulled the bike over backwards in simulation. A forward lean with the feet in front keeps it
seated (about 118 N on the saddle, steady for 10 s).

**Renamed the same day** to the names the team uses: `bike_scan/` (the scan of our blue Puky
bike) is now `puky_bike/`, `bike/` (the green bike from an internet USDZ model) is now
`yellow_bike/`, and `bike_primitive/` is now `primitive_bike/`. The viewer still accepts the
old names (`real_bike`, `bike_scan`, `bike`, `sketchfab`, `bike_primitive`, `primitive`).

**Considered:** One hard-coded pose for one bike (what we had).

## 2026-11-12: Top-level layout: models, experiments, logs, robot

**Decision:** Every simulation model lives in `models/`: `k1/` (pointer to Booster's pinned
K1 model), `bike/` (the bike we used then, with its source USDZ in `source/`), `bike_scan/`
(textured scan of the real bike, was `assets/bike/puky_bike/`), `primitive_bike/` (old
placeholder, kept for comparison), `hand_clips/` (orange add-on clips; their code is
`src/k1bike/hand_clips.py`) and `k1_on_bike/` (seated pose). Combined scenes are
built in code (`src/k1bike/scene.py`), so they always use the current robot and bike. Skills
live in `experiments/<experiment>/` (was `phases/`), with the curated record of each training
run in `runs/runN/`; raw training output goes to `logs/laai_<experiment>/<date>_<run-name>/`
(gitignored; `scripts/train.py` and `play.py` set mjlab's `--log-root` to `logs/`). Real-robot
tools live in `robot/` (was `deploy/`, plus `robot_info.py`). Handover notes and the
credentials template are in `docs/handover/`; the ROS 2 Docker experiment is in
`legacy/ros2_docker/`. Two superseded setup scripts were deleted.

**Why:** Anyone opening the repository should find the robot, the bike and the robot on the
bike in one place, and tell a skill (experiment), one attempt at it (run) and its raw output
(logs) apart.

**Considered:** Copying Booster's K1 XML and meshes into `models/k1/` (duplicates a pinned
dependency and its updates); storing the combined scenes as XML (would go stale when the
robot or bike changes); keeping logs inside each experiment folder (harder to sync or clean).

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
(capsules, cylinders) of `models/yellow_bike/`, not its visual mesh; they need
updating when the bike model changes.

## 2026-11-11: One folder per project phase

(Folders renamed on 2026-11-12: `phases/` is now `experiments/`, `deploy/` is now `robot/`;
see "Top-level layout" above.)

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

**Decision:** The phase 1 walk task (`experiments/walk/task.py`) trains with the observation,
action, PD gains, default pose and joint order of Booster's `k1_walk` deploy task: 690 inputs
(69 per frame x 10 frames), 20 outputs, action scale 0.25. `experiments/walk/export.py` writes a
TorchScript policy that reorders mjlab's layout into booster_deploy's and checks the result
against mjlab's own actions.

**Why:** booster_deploy's existing `LocomotionPolicy` then runs our policy unchanged, in MuJoCo
and on the robot, without forking booster_deploy.

**Result:** Run 2 walks on the robot (see `experiments/walk/README.md`). Robot-side additions live in
`experiments/walk/deploy/laai_walk.py`: speed limits, safety watchdog, smooth handover, ankle damping variant.

**Considered:** mjlab's default observation layout plus a custom policy class in a
booster_deploy fork.

## 2026-11-09: Train with mjlab; runs 1-2 on Google Colab

**Decision:** Train with [mjlab](https://github.com/mujocolab/mjlab). Walk runs 1 and 2 ran
on a Colab GPU (their notebooks are in `experiments/walk/runs/`); new runs use `scripts/train.py`
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
