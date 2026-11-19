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
