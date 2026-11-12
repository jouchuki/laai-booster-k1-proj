# Phase 1: walking

Status 2026-11-11: run 2 walks and turns on the robot with ankle damping 1.5. Ankle-roll
oscillation remains at 0.35-0.4 m/s with 0.6 rad/s turning.

This is the phase overview. Each run's full settings and results are in its notes:
[run 1](runs/run1/notes.md), [run 2](runs/run2/notes.md). Files of this phase:

| File | What |
|---|---|
| `task.py` | mjlab task `LAAI-K1-Walk-Flat` (run 2 settings) |
| `export.py` | Export to booster_deploy's policy format, with equivalence check and video |
| `deploy/laai_walk.py` | booster_deploy tasks `laai_walk` and `laai_walk_ankle_kd15` |
| `runs/run1/`, `runs/run2/` | Notes, task snapshot, notebooks, policy, robot logs |

All times are local time. The robot's clock runs 7 hours ahead, so robot log names show robot
time (for example `stop_20261111_203535` was at 13:35 local). Colab times are UTC (1 hour behind).

## Timeline

| Date | Time | Step |
|---|---|---|
| 2026-11-06 | 16:16-16:24 | Practice: PPO on Walker2d (Mac) |
| 2026-11-06 | 16:31-16:44 | Practice: K1 standing policy in MuJoCo (Mac) |
| 2026-11-09 | | Robot firmware updated to 1.8.1.2 |
| 2026-11-09 | 15:06-16:03 | Run 1 trained (Colab, started 15:15, resumed once at 16:03) |
| 2026-11-09 | 16:14-16:33 | Run 1 exported and copied to the robot |
| 2026-11-09 | after 16:33 | Run 1 robot test: shaking |
| 2026-11-11 | 10:15-11:45 | Run 2 trained (Colab, started 10:23) |
| 2026-11-11 | 11:53-12:00 | Run 2 resumed once, exported, downloaded |
| 2026-11-11 | morning | Run 1 audit and sim stress tests |
| 2026-11-11 | 12:10 | Run 2 policy installed in booster_deploy |
| 2026-11-11 | 12:55-13:07 | Watchdog task file and dry-run script added |
| 2026-11-11 | 13:13-13:18 | Robot dry runs |
| 2026-11-11 | 13:26-13:36 | Robot walking tests 1-4 (ankle kd 1.0) |
| 2026-11-11 | 13:36-13:50 | Robot control test with Booster `k1_walk` |
| 2026-11-11 | 13:50 | Robot walking test 6 (ankle kd 1.5) |

## Setup

| Item | Value |
|---|---|
| Robot | Booster K1, firmware 1.8.1.2 |
| Robot model | `K1_22dof.xml` (booster_assets, pinned commit) |
| Training | mjlab (MuJoCo Warp), RSL-RL PPO, 4096 environments |
| Deployment | booster_deploy (ROS 2 `/low_state`, `/joint_ctrl`), TorchScript on the robot CPU, 50 Hz |

### Policy interface

Matched to Booster's `k1_walk` task, so booster_deploy runs the policy unchanged.

| Item | Value |
|---|---|
| Controlled joints | 20 (legs and arms; head held still), booster_deploy joint order |
| Action | Joint position offset from the default pose, scale 0.25, tracked by PD |
| PD gains, default pose, effort limits | Booster `k1_walk`: legs kp 100 / kd 2.0, ankles kp 65 / kd 1.0 |
| Observation per frame (69) | Base angular velocity (3), projected gravity (3), velocity command (3), joint position (20), joint velocity x0.1 (20), last action (20) |
| History | 10 frames, oldest first (690 inputs) |
| Critic | Also gets privileged simulator state |

### Training settings shared by all runs

| Item | Value |
|---|---|
| Steps per environment per update | 24 |
| Network | MLP 512-256-128, ELU, actor and critic |
| PPO | gamma 0.99, lambda 0.95, clip 0.2, adaptive learning rate (KL target 0.01), observation normalization |
| Command curriculum | Speed range widened at iteration 1500 |

## Runs

| Run | Date | Main change | Training | Booster MuJoCo | Robot |
|---|---|---|---|---|---|
| [1](runs/run1/notes.md) | 2026-11-09 | Baseline | Reward ~101, no falls | Walks | Violent shaking at handover |
| [2](runs/run2/notes.md) | 2026-11-11 | Actuator delay, dynamics randomization, smoothness penalties | Reward ~110, no falls | Walks; survives 60 ms delay | Walks and turns; ankle-roll oscillation, 94 s of walking with ankle kd 1.5 |

## Robot-side features (`deploy/laai_walk.py`, 2026-11-11)

What `laai_walk` adds to Booster's `k1_walk` settings:

| Feature | Detail |
|---|---|
| Speed limits | Full stick = 0.5 m/s forward, 0.2 m/s back, 0.2 m/s sideways, 0.6 rad/s turn |
| Safety watchdog | Stops the policy if a joint moves faster than 15 rad/s, the action jumps by more than 2.5 for 3 steps, the average action jump over 0.2 s exceeds 0.8, or the body tilts more than 35 degrees |
| Stop log | Each stop prints the joint and its last 0.2 s, and saves the last 5 s to `logs/stop_<robot time>.npz` |
| Smooth handover | Targets slide over 0.5 s from the previous controller's last command to the policy's |
| One CPU thread | `torch.set_num_threads(1)`: policy step 6.7 ms instead of 13.2 ms on the robot |
| Ankle damping variant | `laai_walk_ankle_kd15`: ankle kd 1.0 → 1.5 (inside the training randomization of x0.6-1.6) |

After a stop the robot enters the exit mode: Booster's walking mode by default, or damping
(limp) with `--exit-mode damping`.

Checked in booster_deploy MuJoCo with the run 2 policy:

| Check | Result |
|---|---|
| False stops, 45 s at every command limit, nominal and with 20 ms delay, 20 ms sensor lag, velocity noise, 40% less damping | None |
| Run 1 shaking under the same stress, stopped by the average-jump check | After 1.8 s and 2.9 s (first three checks alone: 4.3 s and 6.9 s) |
| Handover from Booster's policy: largest target step | 0.50 rad → 0.03 rad |
| Handover: peak leg joint speed in the first second | 3.0 → 0.5 rad/s |

## Names used before 2026-11-11

Logs and notes from runs 1 and 2 use the names of that time.

| Then | Now |
|---|---|
| mjlab task `Mjlab-Velocity-Flat-Booster-K1` | `LAAI-K1-Walk-Flat` |
| Log folder `k1_velocity`, `k1_velocity_v2` | `laai_walk` |
| booster_deploy tasks `k1_my_walk`, `k1_my_walk_ankle_kd15` | `laai_walk`, `laai_walk_ankle_kd15` |
| `k1_task.py`, `export_k1.py`, `k1_my_walk.py` | `task.py`, `export.py`, `deploy/laai_walk.py` (this folder) |
| Exported policy `k1_my_walk.pt` | `policy.pt` (run folder), `laai_walk.pt` (installed) |
