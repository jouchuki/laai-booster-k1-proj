# Walk run 1

- Date: 2026-11-09
- Experiment name (log folder): `k1_velocity`
- Iterations, wall time, GPU: 3000 (about 295M steps), about 50 min, A100 (Colab)
- Files: `task.py` (task used), `notebook.ipynb` (notebook used), `notebook_executed.ipynb` (with training output), `policy.pt` (exported policy)

## Hypothesis

A velocity-tracking policy with Booster's k1_walk interface and light domain randomization transfers to the robot.

## Changes vs previous run

Baseline. Settings that later runs changed:

| Item | Value |
|---|---|
| Rewards | Positive: linear and yaw velocity tracking, upright, pose. Penalties: foot clearance and swing height (target 7 cm), foot slip, soft landing, body angular velocity, angular momentum, joint limits, action rate (-0.1) |
| Domain randomization | Ground friction, CoM offset, encoder bias, observation noise, random pushes |
| Actuator delay | None |

## Simulation results

- Episode length: full 1000 steps (no falls) from about iteration 1066
- Mean reward: about 101 (peak about 120), about 97 after the curriculum step at iteration 1500
- Tracking error: about 0.12 m/s linear, about 0.2 rad/s yaw
- Export test: max action difference below 1e-4; swapped joint mapping detected (2.57e-3)
- Video: 12 s scripted commands, no falls, short steps, low foot lift

## Sim-to-sim (booster_deploy MuJoCo)

- Result: pass, walks and follows commands
- Watchdog trips: no watchdog in this run

## Real robot

| Gate | Pass / fail | Notes |
|---|---|---|
| Booster walking controller (X) | Pass | Communication, state and joint commands OK |
| Handover to trained policy (A) | Fail | Immediate violent high-frequency shaking, aborted |

Deployed without watchdog.

## Conclusion and next step

Interface verified (export test, sim-to-sim), so the failure was attributed to the dynamics gap: no actuator delay, PD and damping mismatch (parallel ankles), no friction, armature or mass randomization, weak smoothness penalty (action rate -0.1), narrow reset distribution vs handover state.

Next: run 2 with actuator delay, dynamics randomization, stronger smoothness penalties, watchdog and staged real-robot tests.
