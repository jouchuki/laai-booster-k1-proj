# <Phase> run N

- Date:
- Experiment name (log folder):
- Iterations, wall time, GPU:
- Log folder: `logs/laai_<experiment>/<date>_runN/`
- W&B: (link from `scripts/run_summary.py`)
- Files: `task.py` (task used), `policy.pt` (exported policy), `dry_run/`, `robot_logs/`

## Hypothesis

What this run should fix or test.

## Changes vs previous run

| Item | Before | Now |
|---|---|---|
| | | |

## Simulation results

Paste the block from `uv run --group train python scripts/run_summary.py logs/laai_<experiment>/<run>`
(means over the last 50 iterations, every reward term and termination cause), then add:

- Export test (max action difference):
- Video (link):
- Reward terms that changed most vs the previous run, and why:

## Sim-to-sim (booster_deploy MuJoCo)

- Result:
- Watchdog trips:

## Real robot

### Dry run

| Time | Report | Notes |
|---|---|---|
| | | |

### Tests

| # | Time | Task | Command reached | Time to stop | Stop |
|---|---|---|---|---|---|
| | | | | | |

## Conclusion and next step
