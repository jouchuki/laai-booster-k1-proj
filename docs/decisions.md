# Decision log

Newest first. One entry per decision: what, why, what else was considered.

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
