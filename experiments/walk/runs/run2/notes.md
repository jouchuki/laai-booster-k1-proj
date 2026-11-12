# Walk run 2

- Date: 2026-11-11
- Experiment name (log folder): `k1_velocity_v2`
- Iterations, wall time, GPU: 4000 (about 390M steps), about 80 min, A100 (Colab); exported `model_4000.pt` from the resumed run folder `2026-11-11_10-53-05` (final run plus one update)
- Files: `task.py` (task used), `notebook.ipynb` (notebook used), `notebook_executed.ipynb` (with training output), `policy.pt` (exported policy), `dry_run/` (robot dry-run logs), `robot_logs/` (robot stop logs)

## Hypothesis

Run 1 shook on the robot because of the dynamics gap. Adding actuator delay, dynamics randomization and stronger smoothness penalties gives a calmer policy that transfers.

## Changes vs previous run

| Item | Run 1 | Run 2 |
|---|---|---|
| Actuator delay | None | 0 to 4 physics steps (0-20 ms) |
| PD gains | Fixed | Kp x0.85 to 1.15, Kd x0.6 to 1.6 |
| Link mass and inertia | Fixed | Pseudo-inertia randomization, about ±5%, main bodies |
| Joint friction | None | 0 to 0.3 Nm |
| Joint armature | Fixed | x0.8 to 1.2 |
| Reset state | Near default | Joint position ±0.1 rad, joint velocity ±0.5 rad/s |
| Gyro noise | ±0.2 | ±0.3 |
| Action rate penalty | -0.1 | -0.2 |
| Action acceleration penalty | None | -0.02 |
| Joint acceleration penalty | None | -2.5e-7 |
| Joint torque penalty | None | -1e-5 |
| Alive bonus | None | +1.0 |
| Foot clearance target | 7 cm | 8 cm |
| Iterations | 3000 | 4000 |

## Simulation results

- Episode length: 990-1000 of 1000
- Mean reward: about 110
- Tracking metric (training): 0.36 m/s linear, 0.50 rad/s yaw
- Export test: max action difference 3.58e-7
- Video: 12 s scripted commands, no falls

## Sim-to-sim (booster_deploy MuJoCo)

- Result: walks; 0.41 m/s achieved at a 0.5 m/s command (Booster `k1_walk`: 0.42)
- Stress (vs run 1): survives 40 ms delay + 20 ms sensor lag + velocity noise + kd x0.6 (run 1 falls); survives 60 ms actuator delay, falls at 80 ms (run 1 falls at both, Booster survives both)
- Watchdog trips: none in 45 s at every command limit

## Real robot (2026-11-11)

Free-standing, no harness, one person catching, `--exit-mode damping`. Task names as used then.

### Dry runs (no motor commands)

| Time | Threads | Policy step median / p99 | Loop max | Report |
|---|---|---|---|---|
| 13:13 | default | 12.7 / 23.0 ms | 97.0 ms | |
| 13:14 | default | 13.2 / 24.0 ms | 36.5 ms | 14 OK, 1 WARN (inference) |
| 13:15 | `OMP_NUM_THREADS=1` | 6.7 / 18.6 ms | 29.1 ms | 14 OK, 1 WARN (inference) |
| 13:18 | not recorded | 10.2 / 22.5 ms | 32.0 ms | |

All dry runs: `/low_state` 484-486 Hz, 22 motors, gravity [0.01, -0.01, -1.0], leg joints within 0.08 rad of the default pose, joint velocity noise std 0.01 rad/s, max action 0.68-0.70, no watchdog stop.

### Walking tests

| # | Time | Task | Command reached | Time to stop | Stop |
|---|---|---|---|---|---|
| 1 | 13:26 | `k1_my_walk` | 0.2 m/s | ~34 s | Joint speed 17.2 rad/s |
| 2 | 13:30 | `k1_my_walk` | 0.3 m/s + 0.42 rad/s turn | | Joint speed 15.8 rad/s |
| 3 | 13:35 | `k1_my_walk` | 0.35 m/s | 7.4 s | Joint speed 16.6 rad/s, left ankle roll |
| 4 | 13:36 | `k1_my_walk` | 0.2 m/s | 9.6 s | Joint speed 15.1 rad/s, left ankle roll |
| 5 | 13:36-13:50 | `k1_walk` (Booster) | | | None, walked cleanly |
| 6 | 13:50 | `k1_my_walk_ankle_kd15` | 0.4 m/s + 0.6 rad/s turn | 93.9 s | Joint speed 17.0 rad/s, left ankle roll |

- Tests 1-2: policy loop 50.0 Hz, mean 20.0 ms, max 31.0-32.9 ms; `/low_state` 481-483 Hz.
- Tests 3-4 (`robot_logs/stop_20261111_2035*`, `_2036*`): ankles still while standing; ankle rolls started oscillating (~13 Hz) 1.5-2 s after walking began, both sides, left larger.
- Test 6 (`robot_logs/stop_20261111_2050*`): walked and turned well. Ankle-roll oscillation (~12 Hz) only at 0.35-0.4 m/s with 0.6 rad/s turn. Left ankle seemed a little stiff.

## Conclusion and next step

Walks and turns on the robot with ankle kd 1.5 (`laai_walk_ankle_kd15`) at moderate speed. The left ankle roll oscillates at 0.35-0.4 m/s with 0.6 rad/s turning; ankle kd 1.6 is the most the training randomization covers.

Next: a run focused on the ankles.
