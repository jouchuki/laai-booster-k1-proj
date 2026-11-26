# Walk the bike in a straight line: simple versus shaped rewards

Both policies completed training and simulation evaluation on Hex on 2026-11-24.
Robot testing is pending.

The K1 sits on the unsupported Puky (the free bike of [sit_free](../sit_free/README.md)) and
pushes it along with its feet at a commanded forward speed, on the line it started on. Three
tasks train on exactly the same scene and command, and differ only in their rewards:

| Task | Rewards |
|---|---|
| `LAAI-K1-Push-Puky-Simple` | 5 terms: only the task |
| `LAAI-K1-Push-Puky-Shaped` | The same 5, plus 10 terms for the gait, the posture and the motor load |
| `LAAI-K1-Push-Puky-Form` | The shaped 15, plus 4 terms for the riding form (not trained yet) |

| File | What |
|---|---|
| `task.py` | The three mjlab tasks |
| `evaluate.py` | The final test, identical for all tasks |
| `runner.py` | Runs that test at the last checkpoint and logs it |
| `export.py` | Export to booster_deploy's policy format (both tasks have the same interface) |

No robot task (`deploy/`) exists yet: add it once a policy passes the evaluation.

## Shared by both tasks

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

**Form** adds to the shaped terms (2026-11-26; untested in training):

| Term | Weight | What it targets |
|---|---|---|
| `trunk_to_bike` | +1.0 | exp(-(yaw / 0.15 rad)^2 - (pitch / 0.25 rad)^2), the trunk's forward axis in the bike's frame against the seated one: facing along the bike, pitched as seated. Roll is left to `trunk_orientation`, because leaning the trunk against the bike's lean is how a rider balances |
| `leg_band` | -0.5 | Each knee 0-4 cm and each foot 0-8 cm outward of its hip, across the bike (seated: 1.3 and 3.4 cm). Free inside the band; squared distance outside it, per 2 cm. Inward risks hitting the bike; wider is splayed |
| `bike_collision` | -1.0 | Any contact between the robot and the bike outside the saddle (1 per step), plus the force as a share of the body weight (at most 1 more). Holding the bar is free: the forearm-fork contacts are excluded because the hands are welded to the grips. A side effect is that the forearms touching the fork elsewhere is not seen either |
| `step_symmetry` | +0.5 | Left and right steps alike and half a step apart, while moving: compares each foot's latest step (air time, mean floor force during stance, range of the hip, knee and ankle pitch over the swing) and when each foot lands as a share of its step period after the other foot's landing (0.5 = alternating). 0 until both feet have stepped within the last 2 s, so a still or one-footed robot earns nothing |

The symmetry term compares whole steps, not instantaneous states, because one foot pushing
harder at a given moment is often the correct balancing move. A mirror-symmetry loss in PPO
(rsl_rl's `symmetry` extension) would be the other way to enforce this; it is not used.

Not used in any task: `alive`, `joint_acc_l2`, `joint_torques_l2`, `bike_tilt_rate` (they
added little in the free-bike runs) and a swing-foot clearance term (mjlab's needs a terrain
height sensor this scene does not have; a dragging foot is already penalized as slip).

## Comparing the two

Their returns are not comparable: the reward sums differ. Compare the task metrics both log
(`Episode_Metrics/*`) and the final evaluation.

| Metric | Meaning |
|---|---|
| `speed_error_m_s`, `bike_speed_m_s`, `bike_distance_m` | Speed against the command; distance along the line at the end of the episode |
| `lateral_deviation_m`, `heading_error_deg` | Distance from the line and angle to it |
| `saddle_load`, `feet_on_floor`, `both_feet_off`, `foot_slip_m_s` | Seat and gait |
| `bike_tilt_deg`, `bike_tilt_deg_max`, `trunk_tilt_deg` | Balance |
| `action_change` | Step-to-step action change: high means shaking |
| `torque_ratio_max`, `torque_above_90` | Largest torque as a share of its limit; share of time above 90% (the robot's stop is 90% for 0.5 s) |
| `trunk_bike_yaw_deg`, `trunk_bike_pitch_deg` | Trunk against the bike, from its seated orientation |
| `knee_out_cm`, `foot_out_cm` | Knee and foot outward of the hip across the bike, mean of both legs |
| `bike_collision`, `bike_collision_force` | Share of time touching the bike outside the saddle; that force as a share of the weight |
| `step_air_asymmetry`, `step_force_asymmetry`, `step_swing_asymmetry_deg`, `step_phase_asymmetry` | Left against right on the latest steps: relative air time and force differences, pitch-range difference, phase distance from 0.5. They hold their last value; `steps_tracked` is the share of time both feet have a current step, so read them with it |

**Final evaluation** (`evaluation.json`, W&B `Evaluation/*`): 32 worlds, one 10 s episode at
each command of 0, 0.2 and 0.4 m/s, deterministic actions, no pushes, no sensor noise, the
nominal saddle; measured after the first 2 s. It reports the metrics above plus steps per
second, for the policy and for zero actions. Zero actions do not move the bike, so a policy
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
