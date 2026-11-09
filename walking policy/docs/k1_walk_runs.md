# K1 walking policy: run 1 and run 2

Velocity-tracking walking policy for the Booster K1, trained in simulation and deployed with booster_deploy.

## Software and hardware

| Item | Value |
|---|---|
| Robot | Booster K1, firmware 1.8.1.2 |
| Robot model | K1_22dof.xml from booster_assets |
| Simulator | mjlab (MuJoCo Warp, GPU) |
| RL library | RSL-RL, PPO |
| Training hardware | Google Colab, NVIDIA A100 |
| Deployment | booster_deploy (ROS 2, /low_state and /joint_ctrl), policy as TorchScript on the robot CPU |

## Policy interface

Matched to Booster's `k1_walk` task in booster_deploy so the exported policy can be dropped in.

| Item | Value |
|---|---|
| Controlled joints | 20 (legs and arms, booster_deploy joint order) |
| Control rate | 50 Hz |
| Action | Joint position offset from default pose, scale 0.25, tracked by PD |
| PD gains, default pose | Same as Booster's k1_walk |
| Observation per frame (69) | Base angular velocity (3), projected gravity (3), velocity command (3), joint position (20), joint velocity x0.1 (20), last action (20) |
| History | 10 frames, frame-major (690 inputs) |
| Critic | Asymmetric: also gets privileged sim state |

## Training setup

| Item | Run 1 |
|---|---|
| Parallel environments | 4096 |
| Steps per env per update | 24 |
| Iterations | 3000 (about 295M env steps) |
| Network | MLP 512-256-128, ELU, actor and critic |
| PPO | gamma 0.99, lambda 0.95, clip 0.2, adaptive LR with KL target 0.01, observation normalization |
| Rewards | Linear and yaw velocity tracking, upright, foot clearance (target 7 cm), feet air time; penalties on action rate (-0.1), joint limits, foot slip |
| Domain randomization | Ground friction, CoM offset, encoder bias, observation noise, random pushes |
| Command curriculum | Speed range widened at iteration 1500 |
| Wall time | About 50 min |

## Run 1 results

### Simulation

- Episodes reached the full 1000 steps (no falls) from about iteration 1066.
- Mean reward about 101 (peak about 120), about 97 after the curriculum step.
- Tracking error about 0.12 m/s (linear) and about 0.2 rad/s (yaw).
- Gait: stable over a 12 s scripted command sequence (stand, forward 0.5, curve, backward 0.2, stand), with short steps and low foot lift.

### Export validation

- The policy was exported to TorchScript with input and output permutations from mjlab's layout (term-major, model joint order) to booster_deploy's layout (frame-major, policy joint order).
- Equivalence test: deploy-format inputs built from sim state gave the same actions as the training policy (max difference below 1e-4).
- Negative test: a deliberately swapped joint mapping gave a difference of 2.57e-3, which the test flagged.
- Sim-to-sim: the exported policy walked and followed commands in booster_deploy's MuJoCo simulator.

### Real robot

- Handover sequence: Booster's walking controller first, then switch to the trained policy.
- Booster's controller ran normally, confirming communication, state estimation and joint commands.
- On switching to the trained policy, the robot immediately showed violent high-frequency shaking, and the test was aborted.

### Diagnosis

Interface errors are ruled out by the export test and sim-to-sim, so the failure is attributed to the dynamics gap:

- No actuator delay in simulation; real latency turns aggressive corrections into oscillation.
- Real PD response and damping differ from the model, especially at the parallel-linkage ankles.
- No randomization of joint friction, armature or link mass, so the policy overfits one dynamics model.
- Weak smoothness regularization (action rate -0.1 only), allowing high-gain, jittery actions.
- Handover state (pose and joint velocity) is outside the narrow reset distribution seen in training.

## Run 2 changes

Same interface, network and PPO settings as run 1.

| Item | Run 1 | Run 2 |
|---|---|---|
| Actuator delay | None | Random 0 to 4 physics steps |
| PD gains | Fixed | Kp x0.85 to 1.15, Kd x0.6 to 1.6 |
| Link mass and inertia | Fixed | Pseudo-inertia randomization, about ±5%, main bodies |
| Joint friction | None | 0 to 0.3 Nm |
| Joint armature | Fixed | x0.8 to 1.2 |
| Reset state | Near default | Joint position ±0.1 rad, joint velocity ±0.5 rad/s |
| Action rate penalty | -0.1 | -0.2 |
| Action acceleration penalty | None | -0.02 |
| Joint acceleration penalty | None | -2.5e-7 |
| Joint torque penalty | None | -1e-5 |
| Alive bonus | None | +1.0 |
| Foot clearance target | 7 cm | 8 cm |
| Iterations | 3000 | 4000 (about 70 min) |

Expected effect: smoother, lower-gain actions that hold up across a range of dynamics, at the cost of slightly lower reward and tracking in simulation.

## Run 2 deployment safeguards

Watchdog in the deployed policy, which stops the policy when tripped:

| Check | Threshold |
|---|---|
| Joint velocity | > 15 rad/s |
| Action jump between steps | > 2.5 for 3 consecutive steps |
| Body tilt | > 35 degrees |

Verified in sim-to-sim: no false trips over 700 steps of Booster's walking policy; injected shaking was stopped within 0.04 s.

Command limits: forward 0.5 m/s, backward 0.2 m/s, lateral 0.2 m/s, yaw 0.6 rad/s.

Test gates, in order: export test, sim video, sim-to-sim without watchdog trips, robot supported in a harness (10 s, zero command), free standing (30 s, zero command), slow forward walking.
