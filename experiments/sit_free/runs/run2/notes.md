# Run 2: free-bike balance from scratch

Hypothesis: compare learning without the fixed-bike policy's prior experience against
the matched pretrained run (run 1).

The actor, critic, optimizer and counters all start fresh. Both runs use
`LAAI-K1-Sit-Puky-Free`, 256 environments, seed 42 and 1,500 new PPO iterations, with the
same rewards, randomization, pushes and final 32-episode evaluation (seed 12345).
See [the experiment description](../../README.md) for physics and reward details.

Submitted on 2026-11-17 as Hex CPU job **54646**; started on `node03`.
W&B: [scratch run](https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/63cnaaev).
Comparison group: `free-bike-20261117T211126Z`.
Raw run: `logs/laai_sit_free/2026-11-17_22-11-41_free-bike-20261117T211126Z-scratch/`.
Manifest and source archive:
`logs/laai_sit_free/comparisons/free-bike-20261117T211126Z/`.

Completed successfully after 8:47:02 (SLURM job 54646, exit 0). Final checkpoint:
`model_1499.pt`, SHA-256
`3ac8c58aed75aa076dfbda95c1e2672b956ca097ea1f1e8aa30a3747b5a0f0bd`.
Evaluation: 32/32 successes, 10.00 s mean duration, no bike/torso failures,
54.32048416137695 mean return. The checkpoint, parameters and evaluation report were
downloaded on 2026-11-18 with a matching SHA-256, and exported to `policy.pt` alongside
the local checkpoint. The deployment-format action comparison had zero maximum error.

Local scratch startup before submission succeeded for
three PPO updates; the full local test suite passed (117 tests). No real-robot test has
been performed.
## Archived training metrics

Full TensorBoard events and compressed SLURM output are in [training/](training/). The [config](params/), [evaluation report](evaluation.json), [training summary](training_summary.json), [actor checkpoint](actor_checkpoint.pt) and [exported policy](policy.pt) are included. The shared [video and source archive](../comparison/README.md) accompany both runs.

Summary of [training event data](training/) (`scripts/run_summary.py`, mean of the last 50 iterations)

- Iterations: 1500, about 526 min
- Mean reward: 50.7 (peak 51.1)
- Episode length: 500 steps
- action_change: 0.964
- bike_speed_m_s: 0.0303
- bike_tilt_deg: 0.604
- bike_tilt_deg_max: 2.52
- feet_on_floor: 0.94
- posture_error_deg: 2.22
- saddle_load: 0.683
- trunk_tilt_deg: 0.918
- trunk_tilt_deg_max: 2.84
- W&B: https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/63cnaaev

| Reward term | Contribution divided by the 10 s horizon |
|---|---|
| saddle_load | 1.94 |
| bike_upright | 0.994 |
| trunk_orientation | 0.991 |
| alive | 0.5 |
| feet_on_floor | 0.47 |
| posture | 0.468 |
| action_rate_l2 | -0.191 |
| action_acc_l2 | -0.0476 |
| arm_effort | -0.0352 |
| joint_acc_l2 | -0.00992 |
| joint_torques_l2 | -0.00236 |
| bike_tilt_rate | -0.000661 |
| failure | 0 |

| Termination | Episodes per reset batch |
|---|---|
| bike_fell | 0 |
| fell_off_seat | 0 |
| time_out | 1.25 |


Termination values above are averaged counts per reset batch, not success/failure fractions.
