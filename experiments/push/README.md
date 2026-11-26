# Walk the bike in a straight line: simple versus shaped rewards

Both policies completed training and simulation evaluation on Hex on 2026-11-24
([results](#runs)). Robot testing is pending. `LAAI-K1-Push-Puky-Simple-Symmetric` and
`LAAI-K1-Push-Puky-Simple-Weight-On-Saddle` (added 2026-11-26) have not been trained yet.

The K1 sits on the unsupported Puky (the free bike of [sit_free](../sit_free/README.md)) and
pushes it along with its feet at a commanded forward speed, on the line it started on. Four
tasks train on exactly the same scene and command, and differ only in their rewards:

| Task | Rewards |
|---|---|
| `LAAI-K1-Push-Puky-Simple` | 5 terms: only the task |
| `LAAI-K1-Push-Puky-Shaped` | The same 5, plus 10 terms for the gait, the posture and the motor load |
| `LAAI-K1-Push-Puky-Simple-Symmetric` | Built on Simple: the same 5, plus 1 term, both feet on the floor for similar shares of time |
| `LAAI-K1-Push-Puky-Simple-Weight-On-Saddle` | Built on Simple: the same 5, with the saddle reward full only from 75% of the body weight on the saddle (Simple: 50%) |

| File | What |
|---|---|
| `task.py` | All four mjlab tasks |
| `evaluate.py` | The final test, identical for all tasks |
| `runner.py` | Runs that test at the last checkpoint and logs it |
| `export.py` | Export to booster_deploy's policy format (all tasks have the same interface) |

No robot task (`deploy/`) exists yet: add it once a policy passes the evaluation.

## Shared by all tasks

| Item | Setting |
|---|---|
| Scene | K1 seated on the Puky, hands welded to the grips. The bike is free: nothing holds it upright or on course, and the steering turns under the robot's hands |
| Policy interface | The walk policy's: 690 inputs, 20 joints, seated default pose. The velocity command slot carries the forward speed; sideways speed and turn rate are always zero |
| Command | 20% zero (sit still); otherwise 0.15-0.3 m/s forward, widened to 0.15-0.6 m/s at iteration 1500. A new command every 3-8 s |
| Straight line | The line the bike starts on. The policy cannot sense its offset from it; the critic can |
| Saddle | Per episode, the saddle's collision surface moves as one piece by up to 2 cm forward or back and 1.5 cm up or down (`SADDLE_SHIFT_X`, `SADDLE_SHIFT_Z`). The visible saddle does not move |
| Episode | 20 s. Ends early when the bike tilts more than 20 deg or the trunk more than 35 deg from seated |
| Pushes, motor and body randomization, torque limits, PPO | As [sit_free](../sit_free/README.md) (pushes of at most 0.2 m/s) |
| Training | 3,000 iterations by default, logged to `logs/laai_push/` and W&B group `laai_push` |

Holding the seated pose with zero actions, the moved saddle alone leans the bike by up to
12 deg (4 deg on average, 32 worlds, 10 s, no pushes) without a fall; with the nominal saddle
it is at most 4 deg.

## Rewards

Weights are per second (mjlab multiplies them by the 0.02 s control step). They are starting
values and have not been tuned.

**Simple** (also part of the shaped task, with the same weights):

| Term | Weight | What it rewards |
|---|---|---|
| `track_bike_speed` | +2.0 | exp(-((command - speed) / 0.15 m/s)^2), on the bike's forward speed averaged over about 0.5 s (each push-off surges and fades) |
| `on_line` | +1.0 | exp(-(offset / 0.3 m)^2 - (heading / 0.3 rad)^2): on the starting line, heading along it |
| `saddle_load` | +1.0 | Seated: full from 50% of the body weight resting vertically on the saddle |
| `failure` | -5 per fall | Episode ended by a fall |
| `action_rate_l2` | -0.2 | Action change between steps |

The 0.15 m/s tolerance is why non-zero commands start at 0.15 m/s: standing still earns 2% of
the tracking reward at a 0.3 m/s command, but 37% at 0.15 m/s.

**Shaped** adds:

| Term | Weight | What it targets |
|---|---|---|
| `feet_air_time` | +0.5 | Real steps instead of shuffling (moving only) |
| `foot_slip` | -0.25 | A foot sliding on the floor while it is down (moving only) |
| `both_feet_off` | -1.0 | Both feet in the air: one foot down is the safety net |
| `feet_on_floor` | +0.5 | Both feet down while the command is zero |
| `posture` | +0.5 | Near the seated pose: strict when still, loose hips, knees and ankles when moving |
| `trunk_orientation` | +1.0 | Trunk as when seated, no rocking |
| `bike_upright` | +1.0 | exp(-(bike tilt / 10 deg)^2) |
| `arm_effort` | -1e-3 | Arm torque: the hands only rest on the bar |
| `action_acc_l2` | -0.02 | Oscillating actions |
| `torque_near_limit` | -1.0 | Torque above 80% of a motor's limit; one motor at its limit costs 1 |

**Simple-Symmetric** is the simple task plus one term, because the simple policy
(`2026-11-24_17-07-08_simple`) pushed with one foot, on its toes, and held the other leg up:

| Term | Weight | What it rewards |
|---|---|---|
| `foot_balance` | +1.0 | exp(-((left share - right share) / 0.3)^2), each foot's share of time on the floor averaged over about 2 s |

One foot held up while the other pushes earns about 0. Alternating steps at one per second earn
about 0.85; both feet down earns 1, so sitting still costs nothing. The term does not ask for
a particular gait (pushing off with both feet together also earns 1) and does not address
tiptoeing.

**Simple-Weight-On-Saddle** is the simple task with one setting changed: `saddle_load` pays in
full only from 75% of the body weight resting vertically on the saddle, instead of 50%. Same
weight (+1.0); below the target it falls linearly to 0 with no weight on the saddle. The
simple run already rested 0.80 of its weight on the saddle in training (0.87 in the final
evaluation), so this tests whether a stricter target changes the gait, not whether the robot
sits down.

Not used in any task: `alive`, `joint_acc_l2`, `joint_torques_l2`, `bike_tilt_rate` (they
added little in the free-bike runs) and a swing-foot clearance term (mjlab's needs a terrain
height sensor this scene does not have; a dragging foot is already penalized as slip).

## Comparing the tasks

Their returns are not comparable: the reward sums differ. Compare the task metrics all four
log (`Episode_Metrics/*`) and the final evaluation.

| Metric | Meaning |
|---|---|
| `speed_error_m_s`, `bike_speed_m_s`, `bike_distance_m` | Speed against the command; distance along the line at the end of the episode |
| `lateral_deviation_m`, `heading_error_deg` | Distance from the line and angle to it |
| `saddle_load`, `feet_on_floor`, `both_feet_off`, `foot_slip_m_s` | Seat and gait |
| `left_foot_down`, `right_foot_down` | Each foot's share of time on the floor; far apart means one leg does the work |
| `bike_tilt_deg`, `bike_tilt_deg_max`, `trunk_tilt_deg` | Balance |
| `action_change` | Step-to-step action change: high means shaking |
| `torque_ratio_max`, `torque_above_90` | Largest torque as a share of its limit; share of time above 90% (the robot's stop is 90% for 0.5 s) |

**Final evaluation** (`evaluation.json`, W&B `Evaluation/*`): 32 worlds, one 10 s episode at
each command of 0, 0.2 and 0.4 m/s, deterministic actions, no pushes, no sensor noise, the
nominal saddle; measured after the first 2 s. It reports the metrics above plus steps per
second and `foot_imbalance` (the gap between the two feet's floor shares, per world, then
averaged), for the policy and for zero actions. Zero actions do not move the bike, so a policy
has learned to push only where its speed error is below the command itself.

## Commands

[docs/commands.md](../../docs/commands.md#train-the-bike-walking-tasks).

## Runs

The final checkpoints are available below. The original checkpoints and `evaluation.json`
are in the listed Hex run directories, relative to the training checkout.

| Task | Final checkpoint | Hex job | Hex run directory |
|---|---|---:|---|
| Simple | [model_2999.pt](runs/simple/model_2999.pt) | 56366 | `logs/laai_push/2026-11-24_17-07-08_simple` |
| Shaped | [model_2999.pt](runs/shaped/model_2999.pt) | 56367 | `logs/laai_push/2026-11-24_17-33-56_shaped` |

Results: training numbers are `scripts/run_summary.py` (mean over the last 50 iterations);
the rest is the final evaluation at a 0.4 m/s command (0.2 m/s in brackets). Mean reward is not
comparable between tasks. Details, the 0 m/s results and conclusions are in each run's notes.

| Run | Date | Task | Iterations | Mean reward | Episode length (of 1000) | Speed error (m/s) | Left / right foot on the floor | Both feet off | Torque above 90% of limit | Booster MuJoCo | Robot | W&B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [simple](runs/simple/notes.md) | 2026-11-24 | Simple | 3000 | 50.1 | 986.8 | 0.065 (0.046) | 0.00 / 0.55 (0.00 / 0.61) | 0.45 (0.38) | 100% of the time | Not done | Not done | - |
| [shaped](runs/shaped/notes.md) | 2026-11-24 | Shaped | 3000 | 112 | 997.6 | 0.015 (0.016) | 0.46 / 0.54 (0.51 / 0.48) | 0.01 (0.02) | 0% | Not done | Not done | - |

Both policies reach the commanded speed without falling. The simple one pushes with its right
foot only, holds its left leg in the air and runs a motor above 90% of its torque limit the
whole time, which would trigger the robot's safety stop. The shaped one alternates both feet
at under half of the torque limits.
