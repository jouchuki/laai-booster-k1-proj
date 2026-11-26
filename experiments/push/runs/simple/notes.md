# Push run `simple`: minimal reward

- Date: 2026-11-24
- Task: `LAAI-K1-Push-Puky-Simple` (5 reward terms, see [the experiment README](../../README.md#rewards))
- Iterations, wall time, GPU: 3,000 from scratch, about 78 min, Hex job 56366 on `batch-gpu` (node08, one RTX 4500), 4,096 environments, seed 42
- Log folder: `logs/laai_push/2026-11-24_17-07-08_simple/` (local copy)
- W&B: not logged to W&B (TensorBoard only)
- Files: [`task.py`](task.py) (task used, identical to the commit it was trained from),
  [`params/`](params/) (env and agent config), [`model_2999.pt`](model_2999.pt) (final checkpoint),
  [`evaluation.json`](evaluation.json) (final evaluation, on Hex),
  [`evaluation_cpu.json`](evaluation_cpu.json) (the same evaluation rerun on a Mac CPU on
  2026-11-26, adding each foot's share of time on the floor). No `policy.pt` export yet.

## Hypothesis

Rewarding only the task (commanded bike speed, staying on the starting line, sitting on the
saddle, not falling, smooth actions) is enough to learn to push the free bike along with the
feet. Compared against the shaped run ([`../shaped/notes.md`](../shaped/notes.md)), trained
at the same time with the same seed, scene and command.

## Changes vs previous run

First push run.

## Simulation results

Summary of `logs/laai_push/2026-11-24_17-07-08_simple` (`scripts/run_summary.py`, mean of the last 50 iterations)

- Iterations: 3000, about 78 min
- Mean reward: 50.1 (peak 65.6)
- Episode length: 986.8 steps
- twist/error_vel_xy: 0.336
- twist/error_vel_yaw: 0.999
- action_change: 1.98
- bike_distance_m: 5.79
- bike_speed_m_s: 0.296
- bike_tilt_deg: 10.7
- bike_tilt_deg_max: 17.1
- both_feet_off: 0.518
- feet_on_floor: 0.241
- foot_slip_m_s: 0.0911
- heading_error_deg: 4.1
- lateral_deviation_m: 0.162
- posture_error_deg: 22.5
- saddle_load: 0.8
- speed_error_m_s: 0.0864
- torque_above_90: 0.987
- torque_ratio_max: 1
- trunk_tilt_deg: 18.5
- trunk_tilt_deg_max: 25.2
- W&B: not logged to W&B

| Reward term | Per second of episode |
|---|---|
| track_bike_speed | 1.67 |
| saddle_load | 0.933 |
| action_rate_l2 | -0.796 |
| on_line | 0.707 |
| failure | -0.00507 |

| Termination | Episodes per reset batch |
|---|---|
| bike_fell | 0.0787 |
| fell_off_seat | 0.00852 |
| time_out | 4.15 |

`twist/error_vel_*` is mjlab's walk metric on the robot's trunk velocity, not used by this
task: the speed that counts is the bike's (`speed_error_m_s`).

### Final evaluation

32 worlds, one 10 s episode per command, no pushes, no sensor noise, nominal saddle, measured
after the first 2 s. From [`evaluation.json`](evaluation.json); foot shares from
[`evaluation_cpu.json`](evaluation_cpu.json) (its other numbers agree within a few percent;
heading error and lateral deviation differ most).

| Command (m/s) | 0 | 0.2 | 0.4 |
|---|---|---|---|
| Success (no fall in 10 s) | 32/32 | 32/32 | 32/32 |
| Bike speed (m/s) | 0.000 | 0.197 | 0.434 |
| Speed error (m/s) | 0.003 | 0.046 | 0.065 |
| Lateral deviation (m) / heading error (deg) | 0.006 / 0.4 | 0.110 / 7.0 | 0.094 / 1.3 |
| Saddle load | 0.88 | 0.87 | 0.87 |
| Left / right foot on the floor | **0.00 / 1.00** | **0.00 / 0.61** | **0.00 / 0.55** |
| Both feet off | 0.01 | 0.38 | 0.45 |
| Touchdowns per second | 0.04 | 4.7 | 3.2 |
| Largest torque / limit | 1.00 | 1.00 | 1.00 |
| Share of time a motor is above 90% of its limit | 0.68 | **1.00** | **1.00** |

Zero actions (the seated pose held) keep every world upright, do not move the bike and stay
at 13% of the torque limit.

### What the policy does

It reaches the commanded speed, but by pushing with the right foot only. The left foot never
touches the floor, not even when the command is to sit still: the left leg is held up in the
air. The right foot pushes off on its toes in short, fast steps (3-5 touchdowns per second),
and both feet are in the air for 40-45% of the time while moving. The trunk leans about 18 deg
from the seated pose and the bike about 11 deg in training. Video:
`logs/laai_push/videos/simple_*mps_20s*.mp4` (local, not in git).

Some motor is above 90% of its torque limit all of the time while moving, and 68% of the time
while sitting still. The robot's safety stop triggers after 0.5 s above 90%, so this policy
would stop immediately on the real robot.

## Sim-to-sim (booster_deploy MuJoCo)

Not done.

## Real robot

Not done, and not planned for this policy (torque limit, see above).

## Conclusion and next step

The minimal reward is enough to make the bike move at the commanded speed, but the gait it
finds is one-legged and runs the motors at their limit. Nothing in the five terms asks for
two feet or for moderate torques, and holding one leg up costs nothing.

Next: `LAAI-K1-Push-Puky-Simple-Symmetric`, the simple task plus one term (`foot_balance`) that
pays for both feet spending similar time on the floor. It tests whether one term fixes the
one-legged gait. It does not address the torque load: if that stays at the limit, the next
single addition to test is the shaped task's `torque_near_limit`.
