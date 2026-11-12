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
project root and stays out of git; link the checkpoint you used from `notes.md`.

| Plan phase | Experiment | Status |
|---|---|---|
| 1. Walking | [walk/](walk/README.md) | Walks and turns on the robot (run 2) |
| 2. Bike model | none: a model, see [models/](../models/README.md) | Measured model, provisional physics |
| 3. Sit on the bike | not created yet; seated scene in [models/k1_on_bike/](../models/k1_on_bike/README.md) | Seated scene and constraints ready, no training task |
| 4-5. Push, ride | not created yet | Not started |

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
   Commands: [docs/commands.md](../docs/commands.md).
4. Add `policy.pt` and the robot logs to the run folder, fill in `notes.md`, and add a row to
   the runs table in `<experiment>/README.md`.

Walk runs 1 and 2 were trained on Google Colab, so their folders also hold the notebook used
(`notebook.ipynb`) and the notebook with its training output (`notebook_executed.ipynb`).
Robot log file names use the robot clock (7 hours ahead of Amsterdam time).
