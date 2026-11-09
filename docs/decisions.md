# Decision log

Newest first. One entry per decision: what, why, what else was considered.

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

## 2026-11-09: K1 walk task settings

**Decision:** The first walking task (`K1Bike-Walk-Flat`, `src/k1bike/tasks/walk.py`) starts
from mjlab's Unitree G1 flat-ground task, with:
- PD gains, effort limits and default pose copied from Booster's `k1_walk` deploy config,
  because `booster_deploy` sends those gains to the real robot.
- The policy controlling 20 joints; the head holds still (as in Booster's `k1_walk`).
- No body linear velocity in the policy's observations: the real K1 cannot measure it.
  The critic (training only) still gets it.

**Still to do for deployment:** `booster_deploy`'s K1 locomotion policy class expects
Booster's observation layout (with a history of past observations). Our policy uses
mjlab's layout, so the `booster_deploy` fork needs a small policy class that builds
mjlab-style observations and runs our exported ONNX.

## 2026-11-09: Train with mjlab

**Decision:** Use [mjlab](https://github.com/mujocolab/mjlab) for training, starting with its
velocity (walking) task adapted to the K1 model from `booster_assets`.

**Why:**
- It uses MuJoCo for both training and testing, so the policy is trained in the same physics
  engine we evaluate in (`booster_deploy --mujoco`) and that our bike model is written for.
- It offers an Isaac Lab-style API (managers for rewards, observations, randomization), so
  examples and papers written for Isaac Lab carry over.
- It runs massively parallel on NVIDIA GPUs via MuJoCo Warp, which fits the cluster.

**Considered:**
- Isaac Lab via `booster_train`: Booster's own pipeline, but it trains in a different simulator,
  and it only has motion-imitation tasks for the K1 so far.
- Isaac Lab via the community `umn-cse-mnri/booster_ws`: has K1 walking, same simulator concern.

**Consequences:**
- Training needs Linux and an NVIDIA GPU (the cluster); Macs only view and test policies.
- mjlab pins its MuJoCo version, so the whole project uses that version (see `uv.lock`).
