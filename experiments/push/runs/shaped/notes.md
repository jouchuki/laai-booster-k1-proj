# Push run `shaped`: minimal reward plus gait, posture and motor terms

- Date: 2026-11-24
- Task: `LAAI-K1-Push-Puky-Shaped` (15 reward terms, see [the experiment README](../../README.md#rewards))
- Iterations, wall time, GPU: 3,000 from scratch, about 79 min, Hex job 56367 on `batch-gpu` (node08, one RTX 4500), 4,096 environments, seed 42
- Log folder: `logs/laai_push/2026-11-24_17-33-56_shaped/` (local copy)
- W&B: not logged to W&B (TensorBoard only)
- Files: [`task.py`](task.py) (task used, identical to the commit it was trained from),
  [`params/`](params/) (env and agent config), [`model_2999.pt`](model_2999.pt) (final checkpoint),
  [`evaluation.json`](evaluation.json) (final evaluation, on Hex),
  [`evaluation_cpu.json`](evaluation_cpu.json) (the same evaluation rerun on a Mac CPU on
  2026-11-26, adding each foot's share of time on the floor). No `policy.pt` export yet.

## Hypothesis

The minimal reward of the simple run plus ten terms for the gait (real steps, no slipping,
one foot always down, calm feet when still), the posture (near the seated pose, trunk upright,
bike upright) and the motors (arm effort, oscillation, torque near the limit) gives a gait
that could run on the robot. Compared against the simple run
([`../simple/notes.md`](../simple/notes.md)), trained at the same time with the same seed,
scene and command.

## Changes vs previous run

| Item | Simple | Shaped |
|---|---|---|
| Reward terms | 5 | The same 5 with the same weights, plus `feet_air_time`, `foot_slip`, `both_feet_off`, `feet_on_floor`, `posture`, `trunk_orientation`, `bike_upright`, `arm_effort`, `action_acc_l2`, `torque_near_limit` |

## Simulation results

Summary of `logs/laai_push/2026-11-24_17-33-56_shaped` (`scripts/run_summary.py`, mean of the last 50 iterations)

- Iterations: 3000, about 79 min
- Mean reward: 112 (peak 124)
- Episode length: 997.6 steps
- air_time_mean: 0.191
- slip_velocity_mean: 0.107
- twist/error_vel_xy: 0.259
- twist/error_vel_yaw: 1.04
- action_change: 1.4
- bike_distance_m: 5.82
- bike_speed_m_s: 0.293
- bike_tilt_deg: 1.44
- bike_tilt_deg_max: 5.32
- both_feet_off: 0.0628
- feet_on_floor: 0.552
- foot_slip_m_s: 0.103
- heading_error_deg: 2.98
- lateral_deviation_m: 0.113
- posture_error_deg: 6.38
- saddle_load: 0.623
- speed_error_m_s: 0.058
- torque_above_90: 0.0185
- torque_ratio_max: 1
- trunk_tilt_deg: 1.79
- trunk_tilt_deg_max: 5.09
- W&B: not logged to W&B

| Reward term | Per second of episode |
|---|---|
| track_bike_speed | 1.76 |
| trunk_orientation | 0.968 |
| bike_upright | 0.966 |
| saddle_load | 0.924 |
| on_line | 0.803 |
| posture | 0.412 |
| action_rate_l2 | -0.405 |
| feet_air_time | 0.342 |
| action_acc_l2 | -0.0988 |
| feet_on_floor | 0.0881 |
| both_feet_off | -0.0626 |
| arm_effort | -0.0532 |
| torque_near_limit | -0.017 |
| foot_slip | -0.00422 |
| failure | -0.00131 |

| Termination | Episodes per reset batch |
|---|---|
| bike_fell | 0.0125 |
| fell_off_seat | 0 |
| time_out | 5.03 |

Mean reward is not comparable with the simple run (different reward sums). `twist/error_vel_*`
is mjlab's walk metric on the robot's trunk velocity, not used by this task: the speed that
counts is the bike's (`speed_error_m_s`).

### Final evaluation

32 worlds, one 10 s episode per command, no pushes, no sensor noise, nominal saddle, measured
after the first 2 s. From [`evaluation.json`](evaluation.json); foot shares from
[`evaluation_cpu.json`](evaluation_cpu.json) (its other numbers agree within a few percent).

| Command (m/s) | 0 | 0.2 | 0.4 |
|---|---|---|---|
| Success (no fall in 10 s) | 32/32 | 32/32 | 32/32 |
| Bike speed (m/s) | 0.000 | 0.205 | 0.400 |
| Speed error (m/s) | 0.001 | 0.016 | 0.015 |
| Lateral deviation (m) / heading error (deg) | 0.002 / 0.9 | 0.025 / 1.3 | 0.082 / 2.1 |
| Saddle load | 0.66 | 0.73 | 0.70 |
| Left / right foot on the floor | 1.00 / 1.00 | 0.51 / 0.48 | 0.46 / 0.54 |
| Both feet off | 0.00 | 0.02 | 0.01 |
| Touchdowns per second | 0.00 | 2.1 | 2.1 |
| Largest torque / limit | 0.33 | 0.41 | 0.46 |
| Share of time a motor is above 90% of its limit | 0.00 | 0.00 | 0.00 |

Zero actions (the seated pose held) keep every world upright, do not move the bike and stay
at 13% of the torque limit.

### What the policy does

It alternates both feet, about one step per foot per second, with one foot almost always on
the floor, and sits still with both feet down when the command is zero. It tracks the speed
3-4 times more closely than the simple run, stays closer to the line, and keeps the trunk
(1.8 deg) and the bike (1.4 deg) nearly upright in training. Torques stay below half of their
limits in the evaluation. In training, with pushes and randomization, a motor is above 90% of
its limit 1.9% of the time. It rests less weight on the saddle than the simple policy
(0.70 against 0.87). Video: `logs/laai_push/videos/shaped_*mps_20s.mp4` (local, not in git).

## Sim-to-sim (booster_deploy MuJoCo)

Not done. Export with `python -m experiments.push.export` first.

## Real robot

Not done. No `deploy/` task exists yet for the push experiment.

## Conclusion and next step

The shaped reward gives a two-footed, alternating gait at moderate torques that passes the
final evaluation at every speed: the candidate for sim-to-sim and the robot. Which of its ten
extra terms matter is not known; `LAAI-K1-Push-Puky-Simple-Symmetric` tests one of them in
isolation, on the gait asymmetry of the simple run.

Before the robot: export `policy.pt`, run it in Booster's MuJoCo, add a push robot task under
`deploy/`, and check the 1.9% of training time above 90% torque (the robot stops after 0.5 s).
