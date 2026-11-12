# Phase 3: sit on the bike

Status 2026-11-12: training task `LAAI-K1-Sit-Puky` set up and smoke-tested on CPU, not
trained yet. The seated K1 is modelled on the Puky
([models/k1_on_bike](../../models/k1_on_bike/README.md)), and a robot task holds the solved
seated pose for the first real-robot tests.

| File | What |
|---|---|
| `task.py` | mjlab training task `LAAI-K1-Sit-Puky` (below) |
| `export.py` | Export a checkpoint to booster_deploy's policy format, with the same check as the walk export |
| `deploy/laai_sit.py` | booster_deploy task `laai_sit_hold`: holds the seated pose on the Puky, no policy, with safety stops |
| `check_hold.py` | Rehearses that task in simulation on the bike, in the same order as on the robot |

## Training task `LAAI-K1-Sit-Puky`

| Item | Setting |
|---|---|
| Scene | K1 seated on the Puky; the bike is fixed to the world (held upright; steering and wheels still turn); hands welded to the grips; floor |
| Start | The solved seated pose (`models/k1_on_bike/puky_bike.json`), joints off by up to 0.05 rad |
| Policy interface | Same as the walk policy and booster_deploy's `k1_walk`: 20 joints, target = seated pose + 0.25 x action, k1_walk PD gains, 690 inputs (69 x 10 frames); the velocity command input is always 0 |
| Rewards | Weight on the saddle (full reward from 60% of the K1's weight), both feet on the floor, trunk in its seated orientation, posture near the seated pose, alive; penalties on arm effort (hands only rest on the bar), action rate and acceleration, joint acceleration and torque |
| Ends early | Trunk more than 35 deg from its seated orientation |
| Randomization | Motor gains, body mass, joint friction, actuator delay (as walk run 2), pushes every 2-4 s |
| Episode | 10 s; PPO as the walk task, 1500 iterations, log folder `logs/laai_sit/` |

Holding the seated pose with zero actions already sits (about 85% of the weight on the saddle,
feet on the floor): the policy learns to keep it so when pushed and when the robot differs from
the model. Phase 4 builds on this task: free the bike's roll and use the velocity command.

Train, watch and export: [docs/commands.md](../../docs/commands.md#train-the-sit-task-phase-3).

## What `laai_sit_hold` does, and does not do

It holds every joint at the solved seated pose (`models/k1_on_bike/puky_bike.json`) with
Booster's k1_walk PD gains, the gains the simulation uses. **It does not balance.** The robot
stays up only through its two feet and the bike's two wheels, like someone sitting still on a
balance bike with both feet down. A push that moves its weight outside them tips it over;
active balance comes from the trained policies of phases 3-5.

| Step | What happens |
|---|---|
| X | booster_deploy's standing preparation slides all joints from where they are to the seated pose in 1 s (prepare pose = seated pose, k1_walk gains). Booster's walking mode is never entered. |
| A | The task starts and holds the pose (1 s handover from the last command). |
| Stop or Ctrl+C | The robot enters **damping** (limp) mode, the task's default exit. Never pass `--exit-mode walking`: the walking controller would try to stand and walk off the bike. |

Safety stops, each printing the reason and saving the last 5 s to `logs/stop_<time>.npz`:

| Stop | Limit |
|---|---|
| Trunk tilts away from its seated orientation (the bike is falling, or the robot slides off) | 15 deg |
| Any joint moves fast (holding should be nearly still) | 4 rad/s |
| Any joint's torque stays near its limit (fighting the clips or the bike) | 90% for 0.5 s |
| Any joint stays far from its target | 0.3 rad for 1 s |

## Rehearse it in simulation first

booster_deploy's own MuJoCo mode loads the robot without the bike, so it cannot show this task.
`check_hold.py` runs the same task file in our K1 + Puky scene, with booster_deploy's control
law: robot placed with every joint off by up to `--offset` rad, a helper holding the bike
during X, then the hold.

Linux and macOS, from the repository root (headless report):

```bash
uv run --group sim python experiments/sit/check_hold.py
```

```bash
uv run --group sim python experiments/sit/check_hold.py --offset 0.15 --push 60
```

With the viewer, macOS host:

```bash
uv run --group sim mjpython experiments/sit/check_hold.py --viewer
```

With the viewer, Linux host:

```bash
uv run --group sim python experiments/sit/check_hold.py --viewer
```

Results on 2026-11-12 (Puky, 6 s hold, weight as a share of the K1's 193 N):

| Start | Saddle | Feet | Hands | Largest tilt | Stop |
|---|---|---|---|---|---|
| Exactly in the pose | 84% | 11% | 4% | 4 deg | none |
| Joints off by up to 0.1 rad | 68% | 27% | 5% | 5 deg | none |
| Joints off by up to 0.25 rad | 45-60% | 30-51% | 4-10% | 9 deg | none (torque briefly at its limit during X) |
| 40 N sideways push | 51% | 45% | 4% | 10 deg | none |
| 60 N / 100 N push | | | | 15-16 deg | tilt stop after 3.3 s (the bike starts falling) |

The further the robot is placed from the pose, the more weight ends up on its feet: placement
matters, and the dry run measures it (step 3 below).

## Robot test procedure

People: one holds the bike upright the whole time (the robot does not balance), one stays
ready to hold the robot, one runs the commands. Use a harness if available. Clear the area.

1. **Install and copy (Linux or macOS host)**, with the robot on the network:

   ```bash
   bash robot/setup_robot.sh ROBOT_IP sit
   ```

   This installs `laai_sit_hold`, the seated poses and the dry-run script into booster_deploy
   and copies it to the robot.

2. **Place the robot.** Switch the robot to damping (limp) mode, then lift it onto the Puky:
   pelvis in the lowest part of the saddle, feet flat on the floor in front of the saddle and
   hip-width apart, hands resting on the grips. **First test: hands not clipped**, so the arms
   cannot fight clips that sit elsewhere than the simulation assumes.

3. **Dry run (no motor commands)**, on the robot after `ssh booster@ROBOT_IP`:

   ```bash
   cd ~/booster_deploy && source .venv/bin/activate
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   python3 scripts/dry_run.py --task laai_sit_hold --seconds 10 | tee logs/dry_run_report_$(date +%Y%m%d_%H%M%S).txt
   ```

   Read the line **"largest offsets from the policy's default pose"**: for this task these are
   the offsets from the seated pose. Adjust the robot's placement until every leg joint is
   within about 0.15 rad and every arm joint within about 0.3 rad. If that is impossible, the
   solved pose does not fit the real bike: stop and re-measure. In a dry run the robot does not
   follow the targets, so the watchdog line can FAIL on the torque or tracking stop; that is
   expected here, the offsets are what count.

4. **Hold the pose**, on the robot:

   ```bash
   python3 scripts/deploy.py --task laai_sit_hold
   ```

   | Key / remote | Action |
   |---|---|
   | `x` / X | Joints slide to the seated pose in 1 s (keep holding the bike) |
   | `r` / A | Start holding the pose |
   | Ctrl+C | Stop: damping mode, the robot goes limp (hold it) |

   After A, the bike holder loosens their grip gradually but keeps their hands at the bike.
   Hold 10-30 s, then Ctrl+C and support the robot.

5. **Keep the logs (Linux or macOS host)** in a folder for the test date:

   ```bash
   mkdir -p experiments/sit/robot_tests/YYYY-MM-DD
   ```

   ```bash
   scp 'booster@ROBOT_IP:~/booster_deploy/logs/dry_run_*' 'booster@ROBOT_IP:~/booster_deploy/logs/stop_*' experiments/sit/robot_tests/YYYY-MM-DD/
   ```

   Write a short `notes.md` there: placement offsets from the dry run, whether X and A were
   smooth, any stop and its reason, and how the weight felt (saddle, feet, hands).

Next: hand clips on (once step 3 shows the hands reach the grips with small arm offsets), then
the sit training task.
