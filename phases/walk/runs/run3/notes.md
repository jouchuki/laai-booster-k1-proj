# Walk run 3: no self-collision (ablation of run 2)

- Date: not trained yet (set up 2026-11-11)
- Task: `LAAI-K1-Walk-Flat-NoSelfCollision` (in `../../task.py`; `task.py` here is the snapshot)
- Experiment name (log folder): `laai_walk`, run name `run3-no-self-collision`
- Iterations, wall time, GPU: planned 4000 iterations (as run 2), 4096 environments
- Files: `task.py` (task used), `policy.pt` (exported policy), `dry_run/`, `robot_logs/`

This is an ablation for comparison, not the next robot candidate. David's planned
ankle-focused run is separate; renumber this folder if that run should be run 3.

## Hypothesis

Booster's K1 model lets every collision geom touch every other one: in run 2, MuJoCo
checks **174 self-collision geom pairs** per step (knees, thighs, forearms against the
trunk, and so on). mjlab's own humanoid tasks train without self-collision. Turning
it off should:

1. train faster (fewer collision checks per step), and
2. walk as well as run 2 in simulation (episode length, reward, tracking error),

but it may also learn gaits whose legs or arms would hit each other. Booster's MuJoCo
and the real robot both have self-collision, so this run is only useful if it also
walks there. The result decides the collision rule for the sit phase, where the bike
must collide with the robot anyway (the robot geoms need names and explicit rules).

## Changes vs previous run

Only the collision rule. Everything else is run 2: rewards, randomization, actuator
delay, observations, PPO, curriculum.

| Item | Run 2 | Run 3 |
|---|---|---|
| Collision geom names | Unnamed, except the feet | `<body>_collision<i>` (feet unchanged) |
| Robot-robot contacts | On: 174 geom pairs | Off: 0 pairs (contype 0, conaffinity 1) |
| Robot-ground contacts | 20 geoms | 20 geoms (unchanged) |
| condim, priority, friction | 3, 0, 1.0 (feet 0.8) | Same |

Checked when set up: the default task `LAAI-K1-Walk-Flat` still builds byte-identical
model XML and configs after the K1 settings moved to `src/k1bike/robots/k1.py`, and both
compiled scenes have the same ground contacts and contact parameters.

## How to run the comparison

One PPO seed is noisy, so train run 2's settings again with the same seed, iterations
and command line. The Colab run 2 policy is not a like-for-like baseline (notebook,
one resume).

Linux GPU cluster node or NVIDIA host, from the repository root:

```bash
uv run --group train python scripts/train.py LAAI-K1-Walk-Flat-NoSelfCollision \
  --env.scene.num-envs 4096 --agent.max-iterations 4000 --agent.seed 1 \
  --agent.run-name run3-no-self-collision
```

```bash
uv run --group train python scripts/train.py LAAI-K1-Walk-Flat \
  --env.scene.num-envs 4096 --agent.max-iterations 4000 --agent.seed 1 \
  --agent.run-name run3-baseline
```

Repeat both with `--agent.seed 2` if the first pair is close.

To check whether run 3 relies on its legs passing through each other, watch its
checkpoint in run 2's task, which has self-collision on. The policy interface is the
same, so the checkpoint loads in either task.

Linux host:

```bash
uv run --group train python scripts/play.py LAAI-K1-Walk-Flat \
  --checkpoint-file logs/rsl_rl/laai_walk/YOUR_RUN3_RUN/model_3999.pt \
  --device cpu --num-envs 1 --viewer native
```

macOS host:

```bash
uv run --group train mjpython scripts/play.py LAAI-K1-Walk-Flat \
  --checkpoint-file logs/rsl_rl/laai_walk/YOUR_RUN3_RUN/model_3999.pt \
  --device cpu --num-envs 1 --viewer native
```

Export with `python -m phases.walk.export --task LAAI-K1-Walk-Flat-NoSelfCollision`.

## What to record

| Measure | Run 3 baseline (run 2 settings) | Run 3 (no self-collision) |
|---|---|---|
| Training steps per second | | |
| Wall time for 4000 iterations | | |
| Iteration with full-length episodes | | |
| Mean reward at 4000 | | |
| Tracking error (linear, yaw) | | |
| Falls when played in `LAAI-K1-Walk-Flat` (self-collision on) | | |
| Booster MuJoCo: walks, stress tests from run 2 | | |

## Simulation results

- Episode length:
- Mean reward:
- Tracking error (linear, yaw):
- Export test (max action difference):
- Video (link):

## Sim-to-sim (booster_deploy MuJoCo)

- Result:
- Watchdog trips:

## Real robot

Not planned unless run 3 matches run 2 in Booster's MuJoCo, including the stress tests.

## Conclusion and next step
