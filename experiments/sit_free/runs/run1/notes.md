# Run 1: pretrained seated policy on a free bike

Hypothesis: the fixed-bike policy's seated posture and smooth control provide a useful
initialization for unsupported balance compared with the matched scratch run (run 2).

The entire actor transfers, including observation normalization and exploration
distribution. The critic, optimizer, iteration counter and environment counter start
fresh. Both runs use `LAAI-K1-Sit-Puky-Free`, 256 environments, seed 42 and 1,500 new PPO
iterations. They share reward definitions and the final 32-episode evaluation (seed 12345).
See [the experiment description](../../README.md) for physics and reward details.

Submitted on 2026-11-17 as Hex CPU job **54645**; started on `node03`.
W&B: [pretrained run](https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/6tu6vq9j).
Comparison group: `free-bike-20261117T211126Z`.
Raw run: `logs/laai_sit_free/2026-11-17_22-11-46_free-bike-20261117T211126Z-pretrained/`.

Initialization: fixed-bike run `2026-11-17_17-32-52_cpu-stage0/model_900.pt`, frozen at
submission while that original run continued independently. SHA-256:
`42e0ff4f8d08e1e40cba032d8ac05f4a27c4296007b107053bf25d00318444c1`.
Manifest and source archive:
`logs/laai_sit_free/comparisons/free-bike-20261117T211126Z/`.

Completed successfully after 9:42:25 (SLURM job 54645, exit 0). Final checkpoint:
`model_1499.pt`, SHA-256
`6ac7c0bb1a8261c6b81014ab45412ff5095a629e408c68a38a1ea63e1d438e76`.
Evaluation: 32/32 successes, 10.00 s mean duration, no bike/torso failures,
49.87436294555664 mean return. The checkpoint, parameters and evaluation report were
downloaded on 2026-11-18 with a matching SHA-256, and exported to `policy.pt` alongside
the local checkpoint. The deployment-format action comparison had zero maximum error.

Local checks before submission: 117 tests passed; CPU
startup succeeded from the stage-0 model at iteration 800. A separate 256-environment,
three-update startup test peaked at 1.76 GiB RSS. Its final evaluation had 0/32 successes
and 0.89 s mean survival; this checks the pipeline, not trained performance.
No real-robot test has been performed.
## Archived training metrics

Full TensorBoard events and compressed SLURM output are in [training/](training/). The [config](params/), [evaluation report](evaluation.json), [training summary](training_summary.json), [actor checkpoint](actor_checkpoint.pt) and [exported policy](policy.pt) are included. The shared [video and source archive](../comparison/README.md) accompany both runs.

Summary of [training event data](training/) (`scripts/run_summary.py`, mean of the last 50 iterations)

- Iterations: 1500, about 581 min
- Mean reward: 43.9 (peak 44.5)
- Episode length: 496.3 steps
- action_change: 1.25
- bike_speed_m_s: 0.0283
- bike_tilt_deg: 0.795
- bike_tilt_deg_max: 3.1
- feet_on_floor: 0.833
- posture_error_deg: 9.49
- saddle_load: 0.811
- trunk_tilt_deg: 1.51
- trunk_tilt_deg_max: 3.93
- W&B: https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/6tu6vq9j

| Reward term | Contribution divided by the 10 s horizon |
|---|---|
| saddle_load | 1.9 |
| bike_upright | 0.978 |
| trunk_orientation | 0.97 |
| alive | 0.494 |
| feet_on_floor | 0.416 |
| action_rate_l2 | -0.311 |
| posture | 0.101 |
| action_acc_l2 | -0.0801 |
| arm_effort | -0.066 |
| joint_torques_l2 | -0.0149 |
| joint_acc_l2 | -0.00831 |
| failure | -0.0064 |
| bike_tilt_rate | -0.000576 |

| Termination | Episodes per reset batch |
|---|---|
| bike_fell | 0.00536 |
| fell_off_seat | 0.0139 |
| time_out | 1.3 |


Termination values above are averaged counts per reset batch, not success/failure fractions.
