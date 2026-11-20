# Unsupported bike: pretrained policy versus scratch

Task `LAAI-K1-Sit-Puky-Free` keeps the Puky's six-degree-of-freedom root. The bike can roll,
translate, yaw, pitch and topple. It has no stand hinge, support spring or root damping.
Tyres and the robot's feet contact the floor; the hands remain welded to the grips, as in
the original sit task. This experiment tests seated balance, with zero velocity commands,
not riding toward a target.

Source and run records live here, separately from `experiments/sit/`. Raw output goes to
`logs/laai_sit_free/<date>_<comparison>-pretrained/` and `...-scratch/`. The submission
helper freezes the source checkpoint and a source archive in
`logs/laai_sit_free/comparisons/<comparison>/`, with hashes, commands and job IDs in
`manifest.json`. It never resumes into or overwrites the original fixed-bike run.

## Comparison

Both jobs use 256 environments, seed 42, 1,500 new PPO iterations, 24 steps per environment
per iteration (9,216,000 transitions per run), identical randomization and pushes, and W&B
project `laai-k1`. Each requests 4 CPUs, 4 GB RAM, no GPU, and a 12-hour limit on Hex's
student `batch` partition. The longer limit allows for slower free-body physics; it is a
limit, not a measured runtime. Jobs finish as soon as training and evaluation finish.

The pretrained run imports the entire actor, including its observation normalization and
exploration distribution. The scratch run initializes it randomly. Both start with a new
critic, optimizer, iteration counter and environment counter. The critic now observes the
free bike's gravity direction, angular velocity and linear velocity; the actor still uses
the original 690 inputs and 20 joint actions. Use `--agent.init-policy-from`, not the
full-state resume option `--init-from`, for this comparison.

## Rewards and falls

The original seated posture, feet, trunk and smoothness terms remain. New terms reward
upright bike orientation and penalize roll/pitch angular velocity. Saddle reward counts
vertical support, rather than arbitrary contact-force magnitude, and fades with bike tilt.
Total bike tilt includes roll and pitch; exceeding 20 degrees ends the episode. The original
35-degree trunk limit remains. Any non-timeout failure costs five reward units; a timeout
is not penalized. These thresholds end unsuccessful episodes before the bike lies flat.

In W&B, compare the two runs in their shared `free-bike-<timestamp>` group. Inspect episode
length, `Episode_Termination/bike_fell`, `Episode_Termination/time_out`, bike tilt, saddle
support and smoothness, alongside return. Raw rewards from the old fixed-bike task are
not directly comparable to this task's return.

At the final checkpoint, each job also evaluates deterministic actions on 32 fresh worlds
with seed 12345, observation noise disabled and identical scheduled pushes at 2, 4, 6 and
8 seconds. It measures each world's first episode, up to 10 seconds, and saves
`evaluation.json` plus W&B `Evaluation/*` metrics: success fraction, fall fractions,
episode duration and return. Success means reaching the time limit without a failure.
One training seed per initialization is a pilot comparison, not a statistical conclusion.

Submission and monitoring commands: [docs/commands.md](../../docs/commands.md#compare-free-bike-training-on-hex)
and [cluster/README.md](../../cluster/README.md). No real-robot deployment was validated.

## Completed comparison, 2026-11-18

Both jobs completed 1,500 new updates and saved `model_1499.pt`. Their identical final
32-episode evaluation produced:

| Result | Pretrained actor (run 1) | Scratch (run 2) |
|---|---:|---:|
| Successes | 32/32 | 32/32 |
| Mean episode duration | 10.00 s | 10.00 s |
| Bike/torso falls | 0/0 | 0/0 |
| Mean return | 49.8744 | 54.3205 |
| SLURM elapsed time | 9:42:25 | 8:47:02 |

The scratch policy scored higher on this evaluation; both reached its survival ceiling.
These runs do not establish which initialization generalizes better beyond this one-seed
comparison and the tested disturbances.

Final checkpoints, configs, evaluation reports and verified deployment-format `policy.pt`
exports are downloaded into the corresponding local `logs/laai_sit_free/<run>/` folders.
Both exports matched checkpoint actions exactly in the CPU playback check. A synchronized
10-second local visual trial also passed for both policies. The recording is
`logs/laai_sit_free/comparison-final.mp4`; replay/export commands are in
[the command reference](../../docs/commands.md#compare-the-trained-free-bike-policies-locally).

The [committed artifact index](runs/comparison/README.md) links both runs' complete scalar
histories, compressed console logs, configs, evaluation reports, compact actor checkpoints,
exported policies, original source snapshot and the side-by-side video.

## Analysis, 2026-11-19

**Summary.** Run 2 (scratch) has the better seated posture, and it is the base for the next
experiments. But neither run has learned to balance: the evaluation's pushes are so light
that holding the seated pose with zero actions also passes 32/32, and at stronger pushes zero
actions keep more episodes up than either policy.

### Training

| | Run 1 (pretrained actor) | Run 2 (scratch) |
|---|---|---|
| First iteration with full 10 s episodes | 121 | 358 |
| Mean reward, last 50 iterations | 43.9 (levelled off) | 50.7 (still rising at 1,500) |
| Posture error from the seated pose | 9.5-12 deg; the left foot tucks back (video) | **2.2 deg** |
| Both feet on the floor | 61-83% | **94%** |
| Action change per step (`action_change`) | 1.3-1.8 | **0.96**, falling to the end |
| Saddle load | 81% | 68% (the saddle reward is full from 60%) |
| Falls in the last iterations | a few | none |

Run 2 kept improving to the end: between iterations 800 and 1,500 its posture error fell from
7.5 to 2.1 deg, its action change from 2.6 to 0.9, and feet on the floor rose from 17% to 94%.
More iterations would probably still help.

Run 1 learned to survive quickly, but settled in a worse, restless posture. Its first updates
damaged the imported actor: posture error rose from 3.1 deg at iteration 0 to 17 deg at
iteration 10. The new critic starts random, and the new optimizer still moves every weight in
its first steps. A local check shows the same with run 2 as the start: three updates (8
worlds) changed every actor weight by about 2% and dropped the 0.2 m/s evaluation from 32/32
to 3/32. The observation normalization barely changed.

### What the evaluation measures

Training and evaluation push the robot with at most 0.2 m/s forward and sideways (plus
0.3 rad/s rotation): the simulator sets the robot's speed, as a light bump would. The same
32-world evaluation (seed 12345, pushes at 2, 4, 6 and 8 s), repeated with stronger pushes and
with a zero-action baseline (episodes that reach 10 s without the bike tilting past 20 deg):

| Push | Zero actions | Run 1 | Run 2 |
|---|---:|---:|---:|
| 0.2 m/s (the evaluation) | **32** | 32 | 32 |
| 0.4 m/s | **32** | 17 | 27 |
| 0.6 m/s | **32** | 8 | 17 |
| 0.8 m/s | **22** | 1 | 6 |
| 1.0 m/s | **6** | 0 | 1 |
| 2.0 and 3.0 m/s | 0 | 0 | 0 |

At every push speed the policies do no better than holding still, and they tilt the bike
further (largest tilt at 0.6 m/s: zero actions 9 deg, run 2 13 deg, run 1 17 deg). With
pushes this light there was nothing to learn beyond holding the pose.

### On the robot task (`laai_sit`)

Rehearsed in the K1 + Puky scene with booster_deploy's control law
(`experiments/sit/check_hold.py --task laai_sit`): run 2 sits calmly (62% peak torque, no
stop up to a 40 N push). Run 1 drives the right hip roll to its torque limit within a second
and trips the safety stop, so it must not go to the robot. Details:
[experiments/sit/README.md](../sit/README.md#robot-task-laai_sit-the-trained-policy).

### What to improve

1. **Push harder in training**: the push curriculum below.
2. **Evaluate against the zero-action baseline over a range of pushes**: the final evaluation
   now does this (below). A policy has learned to balance only where it beats the baseline.
3. **Protect an imported actor at the start**: warm the critic up with the actor frozen (or
   a much smaller learning rate) for the first iterations, so it does not lose its posture as
   run 1 did. Not implemented yet.
4. **Train longer, preferably on GPU**: run 2 still improved at 1,500 iterations, which took
   9 h on 4 CPUs with 256 worlds.
5. **At least three seeds per variant** before comparing initializations.
6. **Record the fixed-bike stage 0 run** (`cpu-stage0`, whose iteration 900 started run 1) in
   `experiments/sit/runs/`. Archived on 2026-11-20 as [sit run 1](../sit/runs/run1/notes.md).

## Push curriculum: `LAAI-K1-Sit-Puky-Free-Push`

The free-bike task plus a curriculum on the push speed, like stage 1's lean curriculum in
`experiments/sit/task.py`. `LAAI-K1-Sit-Puky-Free` is unchanged, so the comparison above stays
reproducible.

| Item | Setting |
|---|---|
| Push | Every 2-4 s the robot's forward and sideways speed is set to a random value within +-push (rotation as before, +-0.3 rad/s) |
| Range | push from 0.2 m/s (`PUSH_MIN`, the old setting) to 2.0 m/s (`PUSH_MAX`); push = 0.2 + level x 1.8 |
| Steps | After as many finished episodes as there are worlds: level +0.1 if at least 85% ran to the time limit, -0.1 below 50% |
| Light pushes | Stay part of training at every level (uniform within +-push) |
| Logged | `Curriculum/push/level`, `push_mps`, `success_rate` |
| Everything else | As `LAAI-K1-Sit-Puky-Free` (rewards, 20 deg bike and 35 deg trunk limits, randomization) |

Zero actions survive no episode from about 1 m/s on, so a policy that climbs past that level
balances actively.

### Final evaluation with a push sweep

Every free-bike job (both tasks) still writes the original evaluation, and now adds
`push_sweep` to `evaluation.json`: the success fraction at 0.2, 0.4, 0.6, 0.8, 1.0, 1.5 and
2.0 m/s for the policy and for zero actions, and `beats_zero_action_at_mps`, the pushes where
the policy keeps more episodes up. W&B gets `Evaluation/push_<speed>mps/success` and
`.../zero_action`. On 4 CPUs the sweep adds a few minutes at the end of a job.

### Commands

Train on Hex, starting from run 2's actor (`--agent.init-policy-from`, as run 1 did). The
compact `actor_checkpoint.pt` holds the complete actor:

```bash
sbatch --job-name=sit-free-push --time=24:00:00 cluster/train.slurm LAAI-K1-Sit-Puky-Free-Push --agent.init-policy-from experiments/sit_free/runs/run2/actor_checkpoint.pt --agent.max-iterations 3000 --agent.run-name push-run3
```

CPU only (256 worlds, as the comparison):

```bash
sbatch --job-name=sit-free-push --time=24:00:00 cluster/train_cpu.slurm LAAI-K1-Sit-Puky-Free-Push --env.scene.num-envs 256 --agent.init-policy-from experiments/sit_free/runs/run2/actor_checkpoint.pt --agent.max-iterations 3000 --agent.run-name push-run3
```

From scratch: leave out `--agent.init-policy-from`. Until the critic warm-up (point 3 above)
exists, expect run 2's posture to degrade at first, as run 1's did; a scratch run next to it
shows whether starting from run 2 helps.

Smoke test, Linux or macOS (3 iterations, then the final evaluation with the sweep):

```bash
uv run --group train python scripts/train.py LAAI-K1-Sit-Puky-Free-Push --gpu-ids None --env.scene.num-envs 8 --agent.max-iterations 3 --agent.logger tensorboard --agent.init-policy-from experiments/sit_free/runs/run2/actor_checkpoint.pt
```

Follow `Curriculum/push/push_mps` in W&B: the run works if it climbs past about 1 m/s and
`beats_zero_action_at_mps` is not empty at the end.

### Submitted comparison, 2026-11-20

[Run 3](runs/run3/notes.md), initialized from run 2, and [run 4](runs/run4/notes.md),
from scratch, are running as CPU jobs 55245 and 55246 with 256 worlds and 3,000 updates
each. The [comparison record](runs/push-comparison/README.md) includes the source revision,
submission manifest, acceptance criterion and next-lab requirements. Both have a 24-hour
limit to accommodate the longer training; final results are pending.
