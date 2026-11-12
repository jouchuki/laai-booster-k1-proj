# Phases: one folder per phase of the plan

Everything that belongs to one skill lives in its phase folder. Shared tools stay outside
(`scripts/`, `deploy/`, `src/k1bike/`, `docs/commands.md`).

```
phases/
├── README.md              this file
├── RUN_TEMPLATE.md        notes template for a new training run
├── __init__.py            imports every phase's task (registers them with mjlab)
└── <phase>/               for example walk/
    ├── README.md          phase report: status, timeline, setup, runs, robot-side features
    ├── task.py            mjlab training task (registered when phases/ is imported)
    ├── export.py          export a checkpoint to the robot's policy format (python -m phases.<phase>.export)
    ├── deploy/            booster_deploy task files, installed by deploy/install.sh <phase>
    └── runs/runN/         one folder per training run
        ├── notes.md       hypothesis, changes, simulation and robot results
        ├── task.py        snapshot of ../../task.py as trained
        ├── policy.pt      exported policy
        ├── dry_run/       robot dry-run logs
        └── robot_logs/    robot stop logs (logs/stop_*.npz from booster_deploy)
```

| Phase | Folder | Status |
|---|---|---|
| 1. Walking | [walk/](walk/README.md) | Walks and turns on the robot |
| 2. Bike model | `assets/bike/` (a model, not a policy) | Measured model, provisional physics |
| 3. Sit | not created yet; the seated scene is in `src/k1bike/seat.py` | Seated scene and constraints ready, no training task |
| 4-5. Push, ride | not created yet | Not started |

## Names

Every phase uses the project prefix and its folder name, so it never clashes with Booster's
own tasks: mjlab task `LAAI-K1-<Phase>-...`, log folder and booster_deploy task
`laai_<phase>`, installed policy `laai_<phase>.pt`.

## New phase

1. Create `phases/<phase>/` with `task.py`, `export.py`, `deploy/` and `README.md`
   (copy the structure of `walk/`).
2. Import its task in `phases/__init__.py`.
3. Add a test in `tests/` and a row to the table above and to the plan in the root README.

Every phase uses the K1 from `src/k1bike/robots/k1.py` with its self-collision on, and a bike
the robot collides with (`k1_seated_bike_scene`); see [decisions](../docs/decisions.md).

## New training run

1. Copy `RUN_TEMPLATE.md` to `<phase>/runs/runN/notes.md`; fill in hypothesis and changes.
2. Change `<phase>/task.py`, commit it, and copy it to `<phase>/runs/runN/task.py`.
3. Train (`scripts/train.py`), export (`python -m phases.<phase>.export`), test in Booster's
   MuJoCo and on the robot. Commands: [docs/commands.md](../docs/commands.md).
4. Add `policy.pt` and the robot logs to the run folder, fill in `notes.md`, and add a row to
   the runs table in `<phase>/README.md`.
5. Checkpoints and videos stay out of git (link them from `notes.md`).

Walk runs 1 and 2 were trained on Google Colab, so their folders also hold the notebook used
(`notebook.ipynb`) and the notebook with its training output (`notebook_executed.ipynb`).
Robot log file names use the robot clock (7 hours ahead of Amsterdam time).
