# Run 1: fixed-bike CPU stage 0

Trained `LAAI-K1-Sit-Puky` on Hex on 2026-11-17 as job **54630**, on `node03`.
The bike was fixed upright and both hands welded to the grips. This run teaches the
seated posture; it does not demonstrate balance on a free bike.

- 256 environments, seed 42, 1,500 PPO iterations, 24 steps per environment per update.
- 4 CPUs, 4 GB requested RAM, no GPU; SLURM completed with exit `0:0` in **7:32:41**.
- Raw output on Hex: `/scratch/laoa/laai-booster-k1-proj/logs/laai_sit/2026-11-17_17-32-52_cpu-stage0/`.
- Training source: commit `7ade19e171a6cf575df9364e6ea8088aa2096bf6` plus the archived
  `training/laai-booster-k1-proj.diff.gz`. `task.py` is the unchanged sit task from that commit.
- [Provenance and full-checkpoint hashes](provenance.json), [parameters](params/),
  [training curves and compressed console log](training/), [training summary](training_summary.json).

The final `model_1499.pt` was downloaded and its actor is retained in
`actor_checkpoint.pt`. The exported `policy.pt` was checked locally on Linux in the
fixed-bike environment for 10 seconds: **maximum action difference 0.00, no episode reset**.
This verifies the export format, not real-robot performance. No real-robot test of this
fixed-bike policy is recorded.

The intermediate `model_900.pt` supplied the actor for free-bike run 1. Its SHA-256 is
`42e0ff4f8d08e1e40cba032d8ac05f4a27c4296007b107053bf25d00318444c1`, matching the
[free-bike comparison manifest](../../../sit_free/runs/comparison/manifest.json).
Its compact actor is retained separately as `actor_checkpoint_900.pt`.

## Simulation results

Summary of `/home/jouchuki2/study_vu/laai/laai-booster-k1-push-20261120/logs/laai_sit/2026-11-17_17-32-52_cpu-stage0` (`scripts/run_summary.py`, mean of the last 50 iterations)

- Iterations: 1500, about 452 min
- Mean reward: 43.8 (peak 43.9)
- Episode length: 500 steps
- action_change: 0.5
- feet_on_floor: 0.993
- posture_error_deg: 1.61
- saddle_load: 0.728
- trunk_tilt_deg: 0.542
- trunk_tilt_deg_max: 1.73
- W&B: https://forge.coreweave.com/wandb/K1Booster/laai-k1/runs/ckhliij3

| Reward term | Per second of episode |
|---|---|
| saddle_load | 1.99 |
| trunk_orientation | 0.997 |
| alive | 0.5 |
| feet_on_floor | 0.496 |
| posture | 0.482 |
| action_rate_l2 | -0.0516 |
| arm_effort | -0.0156 |
| action_acc_l2 | -0.0129 |
| joint_acc_l2 | -0.00247 |
| joint_torques_l2 | -0.00101 |

| Termination | Episodes per reset batch |
|---|---|
| fell_off_seat | 0 |
| time_out | 1.29 |
