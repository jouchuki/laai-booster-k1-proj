# Experiments: one folder per skill

Each experiment is one skill from the project plan (walk, sit, push, ride). Its folder holds
everything for that skill: the training task, the export to the robot, the robot-side task
files, and a record of every training run. Shared code and models stay outside
(`src/k1bike/`, `models/`, `scripts/`, `robot/`, `docs/commands.md`).

```
experiments/
├── README.md              this file
├── RUN_TEMPLATE.md        notes template for a new training run
├── __init__.py            imports every experiment's task (registers them with mjlab)
└── <experiment>/          for example walk/
    ├── README.md          experiment report: status, timeline, setup, runs, robot-side features
    ├── task.py            mjlab training task (registered when experiments/ is imported)
    ├── export.py          checkpoint -> robot policy (python -m experiments.<experiment>.export)
    ├── deploy/            booster_deploy task files, installed by robot/install.sh <experiment>
    └── runs/runN/         record of one training run (kept in git)
        ├── notes.md       hypothesis, changes, simulation and robot results
        ├── task.py        snapshot of ../../task.py as trained
        ├── policy.pt      exported policy
        ├── dry_run/       robot dry-run logs
        └── robot_logs/    robot stop logs (logs/stop_*.npz from booster_deploy)

logs/laai_<experiment>/<date>_<run-name>/   raw training output (gitignored, project root)
```

**Run record vs. training logs.** `runs/runN/` is the curated record that stays in git. The
raw training output (checkpoints, TensorBoard, videos) goes to `logs/laai_<experiment>/` at the
project root and stays out of git; link the checkpoint you used from `notes.md`. Curves and
configs are also in W&B; see [Tracking and comparing runs](#tracking-and-comparing-runs).

| Plan phase | Experiment | Status |
|---|---|---|
| 1. Walking | [walk/](walk/README.md) | Walks and turns on the robot (run 2) |
| 2. Bike model | none: a model, see [models/](../models/README.md) | Measured model, provisional physics |
| 3. Sit on the bike | [sit/](sit/README.md); seated scene in [models/k1_on_bike/](../models/k1_on_bike/README.md) | Training task `LAAI-K1-Sit-Puky` trained on Hex (`cpu-stage0`), not yet recorded here; robot task `laai_sit_hold` keeps the K1 seated on the robot with the bike held (2026-11-19); robot task `laai_sit` runs a trained policy, rehearsed in simulation only |
| 3. Unsupported balance | [sit_free/](sit_free/README.md) | Pretrained (run 1) and scratch (run 2) both pass the 0.2 m/s evaluation 32/32, but so do zero actions, which do better under stronger pushes ([analysis](sit_free/README.md#analysis-2026-11-19)). Run 2 has the better posture and is the base for later experiments; run 1 trips the hip-roll torque stop in the robot task. Push-curriculum task `LAAI-K1-Sit-Puky-Free-Push` ready, not trained |
| 4-5. Walk the bike | [push/](push/README.md) | Tasks `LAAI-K1-Push-Puky-Simple` and `-Shaped` ready (free bike, forward-speed command, straight line, saddle randomization); not trained |

## Names

Every experiment uses the project prefix and its folder name, so it never clashes with
Booster's own tasks: mjlab task `LAAI-K1-<Experiment>-...`, log folder and booster_deploy task
`laai_<experiment>`, installed policy `laai_<experiment>.pt`.

## New experiment

1. Create `experiments/<experiment>/` with `task.py`, `export.py`, `deploy/` and `README.md`
   (copy the structure of `walk/`).
2. Import its task in `experiments/__init__.py`.
3. Add a test in `tests/` and a row to the table above and to the plan in the root README.

Every experiment uses the K1 from `src/k1bike/robots/k1.py` with its self-collision on, and a
bike the robot collides with (`k1_seated_bike_scene`); see [decisions](../docs/decisions.md).

## New training run

1. Copy `RUN_TEMPLATE.md` to `<experiment>/runs/runN/notes.md`; fill in hypothesis and changes.
2. Change `<experiment>/task.py`, commit it, and copy it to `<experiment>/runs/runN/task.py`.
3. Train (`scripts/train.py ... --agent.run-name runN`), export
   (`python -m experiments.<experiment>.export`), test in Booster's MuJoCo and on the robot.
4. Run `scripts/run_summary.py` on the log folder. Paste its block into `notes.md` and its row
   into the experiment README's Runs table, then fill in the change, sim-to-sim and robot cells.

## Tracking and comparing runs

Each layer has one job:

| Layer | Holds | Where |
|---|---|---|
| Raw output | Checkpoints, TensorBoard files, videos | `logs/laai_<experiment>/<date>_<run-name>/` (gitignored) |
| Live tracking | Curves, every reward term, terminations, task metrics, config, git diff, videos | W&B project `laai-k1`, one group per experiment, tagged with the task id |
| Curated record | Hypothesis, changes, sim-to-sim and robot results, conclusion | `<experiment>/runs/runN/notes.md` and the Runs table in `<experiment>/README.md` (in git) |

Use one name for a run everywhere: `--agent.run-name runN` names the log folder, the W&B
run and `runs/runN/`. To compare runs, select them in W&B (filter by group) and open the
`Episode_Reward/*` panels to see which reward terms moved. The config diff shows what
changed. Each Runs table holds the same headline numbers for every run. From now on they
come from `scripts/run_summary.py` (mean over the last 50 iterations), so runs can be
compared row by row.

If a run fails on the robot but looks fine in simulation, record it in `notes.md`: W&B only
sees training.
   Commands: [docs/commands.md](../docs/commands.md).
4. Add `policy.pt` and the robot logs to the run folder, fill in `notes.md`, and add a row to
   the runs table in `<experiment>/README.md`.

Walk runs 1 and 2 were trained on Google Colab, so their folders also hold the notebook used
(`notebook.ipynb`) and the notebook with its training output (`notebook_executed.ipynb`).
Robot log file names use the robot clock (7 hours ahead of Amsterdam time).
