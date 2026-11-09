# K1 walk

Training a walking policy for the Booster K1 in simulation (mjlab, PPO) and deploying it on the real robot with booster_deploy. First step towards teaching the K1 to ride a bike.

## Runs

| Run | Date | Main change | Sim reward | Sim-to-sim | Real robot | Result |
|---|---|---|---|---|---|---|
| [run1](runs/run1/notes.md) | 2026-11-09 | Baseline | ~101 | Pass | Shook on handover | Dynamics gap |
| [run2](runs/run2/notes.md) | | Actuator delay, dynamics randomization, smoothness penalties, watchdog | | | | |

Full technical write-up: [docs/k1_walk_runs.md](docs/k1_walk_runs.md)

## Layout

| Path | What |
|---|---|
| `k1_task.py` | mjlab task (current version) |
| `train_k1.py` | Training entry point |
| `export_k1.py` | Export to TorchScript for booster_deploy, with equivalence test and video |
| `deploy/k1_my_walk.py` | booster_deploy task with safety watchdog |
| `deploy/setup_robot.sh` | Copies booster_deploy to the robot and sets up its environment |
| `notebooks/` | Colab training notebook (current version) |
| `runs/runN/` | Per run: notes, exact task file and notebook used |
| `runs/TEMPLATE.md` | Template for new run notes |

## Workflow for a new run

1. Copy `runs/TEMPLATE.md` to `runs/runN/notes.md`, fill in hypothesis and changes.
2. Copy the `k1_task.py` you will train with into `runs/runN/`.
3. Commit and tag: `git tag runN && git push --tags`.
4. Train, export, test. Fill in results in notes.md and the table above.
5. Commit the exported policy as `runs/runN/k1_my_walk.pt`. Videos and checkpoints go to Drive or a GitHub Release, linked from notes.md.
