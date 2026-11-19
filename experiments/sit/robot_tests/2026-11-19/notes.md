# Robot test 2026-11-19: `laai_sit_hold` and `laai_sit` on the Puky

First real-robot runs of `laai_sit_hold` (morning) and of a trained policy, free-bike run 2
under `laai_sit` (afternoon). The hands rested on the grips without clips, and someone held
the bike the whole time. Times below are local; the robot's clock (and so the log names) is 7
hours ahead.

## Result

- **The robot sits by itself on the Puky with the bike held fixed.** Two holds, 28 s and
  32 s, ran at 50 Hz with no safety stop. This is the situation the stage 0 training task
  `LAAI-K1-Sit-Puky` starts from (bike fixed to the world).
- **The real saddle was 2.6 cm higher than the model's.** With it, the solved pose could not
  be reached by placement: hips and knees stayed 0.3-0.6 rad less bent than the pose. Lowered
  to the model's height, the legs fit.
- **Without clips the hands do not sit where the model puts them**, so the arms press down
  on the bar. The left shoulder roll was 0.49 rad off the pose in every dry run.
- **The trained policy (free-bike run 2) sat calmly for at least 35 s**, then a push set the
  ankle rolls oscillating at about 9 Hz until the speed stop, 40 s after start. The left hand
  came loose. Next try: ankle damping 1.5 (`laai_sit_ankle_kd15`), feet on the floor.
  [Details](#policy-test-free-bike-run-2).

## Setup

| Item | Value |
|---|---|
| Robot | Booster K1, firmware 1.8.1.2, robot at 10.0.0.238 |
| Task | `laai_sit_hold` (no policy), installed with `bash robot/setup_robot.sh 10.0.0.238 sit` |
| Bike | Puky, held upright by hand the whole time |
| Hands | Resting on the grips, not clipped |
| People | Bike holder, robot catcher, operator |

Setup problems, both fixed on the day:

- From the VS Code terminal, `ssh` failed with "No route to host": VS Code had no macOS
  Local Network permission. Running from the Terminal app worked.
- `setup_robot.sh` step 3 failed with `Cannot create symlink to 'booster/booster_deploy': File
  exists`: tar copied the `booster_deploy -> booster/booster_deploy` symlink, not the folder.
  The script now resolves the real path first (`pwd -P`).

## Placement: dry runs with the saddle as found (dip 43 cm)

`dry_run.py --task laai_sit_hold`, robot limp, placed by hand. Offsets are measured minus
seated pose, in rad. In a dry run the motors do not move, so the torque and tracking stops in
the FAIL line are expected; the offsets and the would-be torque are what count.

| Local time | Placement | Trunk tilt | Hips L / R | Knees L / R | Would-be torque |
|---|---|---|---|---|---|
| 13:32 | First try | 5.9 deg | +0.61 / - | -0.57 / - | 195% (L hip) |
| 13:33 | Left leg mirrored to the right | 23.6 deg | - | -0.48 / -0.41 | 138% (L knee) |
| 13:35 | Trunk upright | 8.6 deg | +0.47 / +0.51 | -0.41 / -0.40 | 168% (R hip) |
| 13:36 | Feet slid back | 20.1 deg | +0.32 / +0.38 | -0.37 / -0.47 | 134% (R knee) |
| 13:37 | Natural seat, trunk upright (reference) | 2.0 deg | +0.46 / +0.56 | - / -0.63 | 188% (R hip) |
| 13:40 | Looks like the simulation (reference) | 19.1 deg, 17 -> 22 | +0.37 / +0.33 | -0.39 / -0.43 | 130% (R hip yaw) |
| 13:41 | Looks like the simulation, again | 18.4 deg, 16 -> 26 | +0.33 / +0.44 | -0.41 / -0.52 | 152% (R hip) |

"-" means the joint was not among the five largest offsets the report lists. Every run had
the left shoulder roll at -0.47 to -0.49.

However the robot was placed, with the feet on the floor the hips and knees stayed 0.3-0.6 rad
less bent than the pose. Straightening the trunk bent the legs less still, and moving the feet
back tipped the trunk forward. Limp on the saddle, the robot slowly tips forward (17 -> 26 deg
in 5 s at 13:41): it does not stay seated without stiff motors.

## Measurements against the model

| Quantity | Model (`puky_bike` seated pose) | Real |
|---|---|---|
| Saddle dip (lowest point of the saddle top) | 40.4 cm (highest point 42.0 cm) | **43 cm** as found |
| Hip pitch joint above the floor | 43.6 cm | **45.5 cm** (look-alike placement) |
| Hip joint above the saddle dip | 3.2 cm | 2.5 cm |
| Trunk tilt (forward) | 12.9 deg | 2-26 deg, depending on placement |
| Handlebar centre | 53.0 cm | 53 cm |
| Hand centre | 53.0 cm, 4 cm behind the bar centre along a level forearm (clipped) | **56 cm**, on top of the bar (unclipped) |
| Soles | Flat, 5 mm above the floor before settling | Toes on the floor, heels slightly up (after the saddle change) |

The pelvis sits on the saddle the same way in both (hip about 3 cm above the dip); the saddle
itself was higher. Check in the model: with the trunk kept in place, the real hip and knee
angles (about 0.4 and 0.45 rad less bent) move the feet 3.2 cm down, about the measured height
difference.

## After lowering the saddle

The seat post was lowered until the saddle dip was at **40.4 cm**, the model's value.

| Local time | Step | Result |
|---|---|---|
| 13:50 | Dry run | Trunk 10.5 deg. Left hip +0.22, left knee -0.23, others under 0.2. Would-be torque 76% (L hip). Only stop: tracking on the left shoulder roll (-0.49) |
| 13:51-13:53 | `deploy.py`: X, then `a` by mistake instead of `r` | X held the prepare pose for 80 s with no problems. The task never started (`policy_step` count 0). **Feet about 1 cm above the floor** |
| | Saddle lowered about 1 cm more | Final height not measured exactly (about 39.4 cm) |
| 13:56 | Dry run | Trunk 5.8 deg. Hips +0.25 / +0.29, left shoulder roll -0.49, head pitch -0.34. Would-be torque 109% (L hip) in the first second, while the robot was still settling |
| 13:57 | Hold 1: X, `r` after 22 s, Ctrl+C | **28 s hold, 1408 steps at 50.0 Hz (max 33.6 ms), no safety stop**, clean switch to damping |
| 13:59-14:00 | Hold 2: X, `r` after 22 s, Ctrl+C | **32 s hold, 1583 steps at 50.0 Hz (max 30.8 ms), no safety stop**. Feet on the floor with the heels slightly up. The hands press down on the bar. Sits by itself with the bike held fixed |

The task saves data only on a safety stop, so there is no log of the two holds themselves.

## Hand clips

Measured with a clip on:

- **2 cm from the forearm tip to the bar centre**, along the forearm. This matches the model:
  `src/k1bike/seat.py` puts the bar centre 2 cm beyond the forearm tip (`CLIP_HAND_TO_BAR`
  minus `CLIP_HAND_CENTRE_INSET`). The clip offset needs no change.
- **The forearms point straight forward; the bar crosses the hands slightly diagonally.** The
  model takes the clip as collinear with the forearm. The bar's angle across the clip is not
  yet compared with the solved pose.

Why the hands press down without clips: the model holds the hand at bar height, 4 cm behind
the bar, as the clip would. Unclipped, the hand can only rest on top of the bar, 3 cm higher
and further forward, so holding the model's arm angles pushes it down. Moving the handlebar
does not fix this; the clips do.

## Policy test: free-bike run 2

Afternoon: the first trained policy on the robot. `laai_sit` ran free-bike run 2
(`experiments/sit_free/runs/run2/policy.pt`, trained from scratch), installed with
`bash robot/setup_robot.sh 10.0.0.238 sit experiments/sit_free/runs/run2/policy.pt`. The bike
was held, and the hands rested on the grips without clips.

### Dry run (15:34)

| Item | Value |
|---|---|
| Placement | Trunk tilt 9.4 deg. Hip pitch L +0.25 / R +0.35, left hip yaw +0.13, ankle pitch -0.15 / -0.16, left shoulder roll -0.47 (as in every run), head pitch -0.31 |
| Policy actions | Max 1.57-1.72 (warning above 1.6), mean 0.46, steady from the first second. Largest on the arms: left shoulder pitch -1.58 (asks -0.40 rad), right shoulder roll +1.36, right shoulder pitch -0.97, left elbow pitch +0.88. Legs about 0.1 rad |
| Stop (would-be) | Left hip yaw over 90% of its torque limit for 0.5 s: from its 0.13 rad placement offset, which X removes |
| Timing | Inference 10.7 ms median, 20.6 ms at the 99th percentile; one 107 ms loop |

Two reasons a dry run overstates a policy: it skips X, so the policy sees the placement offsets
(training only started within 0.05 rad of the pose), and the motors do not move, so a
requested move never shrinks. Rehearsed in simulation from this exact placement
(`check_hold.py --start-from` with the dry-run log; X, then run 2, bike held): 10 s, no stop,
62% peak torque, 11.4 deg tilt, max action 1.70 (robot: 1.72). Simulation gives the policy the
same inputs as the robot. Even in simulation run 2 pushes its shoulders (left shoulder pitch
-0.17 rad): it learned with the hands welded to the bar.

### Run (15:39-15:40)

`deploy.py --task laai_sit`: X, then `r` 18 s later. The policy ran **40 s** (2000 steps at
50 Hz, longest step 33 ms) until the joint-speed stop: left ankle roll 4.7 rad/s > 4.
Damping followed as intended.

Seen: the feet floated slightly. We pushed the robot a little; the left hand came loose and the
feet started to shake.

From the stop log (`stop_20261119_224032_970.npz`, the last 5 s; seconds after `r`):

| Time | What happened |
|---|---|
| 35.4-39.1 s | Calm: trunk tilt 9 deg, every joint under 0.4 rad/s, actions about 0.8 |
| about 39.1 s | The push: the hips move slightly |
| 39.44 s | The right ankle roll starts swinging |
| 39.4-40.1 s | The swing grows each cycle (0.7, 1.2, 1.6, 2.2, 3.2 rad/s); the left ankle joins |
| 40.1 s | Left ankle roll at 4.7 rad/s: speed stop |

The ankle oscillated, not the policy: the right ankle roll swung over 0.126 rad while its
target moved 0.045 rad smoothly; the oscillation is at least 9 Hz (zero crossings at 50 Hz
sampling) while the action changed slowly (0.17 to 0.32); the trunk stayed at 9.0-9.6 deg.
This is walk run 1's ankle-roll oscillation with ankle damping 1.0
(`experiments/walk/README.md`); `laai_sit` uses 1.0 too. A foot carrying no weight is damped
only by its motor, so the floating feet made it easier to start.

The arms were steady in these 5 s, the left shoulder pitch 0.13-0.19 rad off the pose (the
rehearsal: 0.17). The hand probably came loose before second 35, outside the log.

Simulation does not reproduce the oscillation: in the rehearsal from this placement, with 40
and 60 N pushes, ankle damping 1.0 and 1.5 both run 10 s without a stop.

### Changes after this test

- New task `laai_sit_ankle_kd15`: `laai_sit` with ankle damping 1.5, as
  `laai_walk_ankle_kd15`. Inside the sit training's damping randomization (x0.6 to x1.6).
- For the next run: feet on the floor (lower the saddle a few millimetres; after X check that
  both soles touch), no pushes until it sits calmly for 30 s, a hand near the left grip.

## Problems found in the procedure

- **Our safety stops only run after `r`.** Between X and `r` only Booster's prepare pose holds
  the robot. That lasted 80 s in the first attempt. Press `r` within a few seconds of X.
- On the keyboard, the remote's A button is `r`. `a` sets the sideways speed.
- `dry_run.py` was started once on the Mac instead of on the robot. It wrote an empty
  `logs/dry_run_report_20261119_132936.txt` there.

## Logs

Robot-clock names, from `~/booster_deploy/logs/` on the robot. In this folder: the policy test's
`dry_run_laai_sit_ros_20261119_223420.npz` (15:34), its would-be stop
`stop_20261119_223411_527.npz`, and the run's stop `stop_20261119_224032_970.npz` (15:40). The
hold tests' logs below are **not yet in this folder**.

| Local time | Dry-run log | Stop log |
|---|---|---|
| 13:32 | `dry_run_laai_sit_hold_ros_20261119_203217.npz` | `stop_20261119_203208_222.npz` |
| 13:33 | `dry_run_laai_sit_hold_ros_20261119_203314.npz` | `stop_20261119_203305_121.npz` |
| 13:35 | `dry_run_laai_sit_hold_ros_20261119_203515.npz` | `stop_20261119_203511_041.npz` |
| 13:36 | `dry_run_laai_sit_hold_ros_20261119_203607.npz` | `stop_20261119_203603_702.npz` |
| 13:37 | `dry_run_laai_sit_hold_ros_20261119_203757.npz` (natural, upright) | `stop_20261119_203752_989.npz` |
| 13:40 | `dry_run_laai_sit_hold_ros_20261119_204021.npz` (look-alike) | `stop_20261119_204017_204.npz` |
| 13:41 | `dry_run_laai_sit_hold_ros_20261119_204123.npz` | `stop_20261119_204119_540.npz` |
| 13:50 | `dry_run_laai_sit_hold_ros_20261119_205006.npz` (saddle 40.4 cm) | `stop_20261119_205003_094.npz` |
| 13:56 | `dry_run_laai_sit_hold_ros_20261119_205609.npz` (saddle about 39.4 cm) | `stop_20261119_205605_472.npz` |

The stop logs come from the dry runs' would-be watchdog; the holds had no stop. Each dry run
also wrote a `dry_run_report_<time>.txt`.

## Next

1. Measure the final saddle dip height. Record the seat-post setting, so the bike can be set
   up the same way next time.
2. Compare the bar's angle across the clip (slightly diagonal on the real robot) with the
   solved pose. If they differ, change the clip orientation in `src/k1bike/seat.py` and run
   `scripts/solve_seated_pose.py` again. The 2 cm offset already matches.
3. Hold test with the hands clipped, once the clip geometry matches the model.
4. Train `LAAI-K1-Sit-Puky`. With the saddle at the model's height, the training scene
   matches the real setup in the legs and trunk.
5. Save a short log of every hold in `laai_sit.py`, not only on a stop, so successful holds
   leave data too.
