# Deploy: shared robot tools

Booster's `booster_deploy` (cloned by `scripts/setup_booster.sh`, gitignored) runs our
policies in MuJoCo and on the robot. Each phase keeps its own booster_deploy task files in
`phases/<phase>/deploy/`; the tools here install and test them for any phase.

Commands for every step (install, Booster's MuJoCo, copy to the robot, dry run, robot test,
controls, logs) are in [docs/commands.md](../docs/commands.md).

| File | What |
|---|---|
| `install.sh <phase> [policy.pt]` | Copies `phases/<phase>/deploy/*.py`, `dry_run.py` and a policy (default: the newest run's `policy.pt`) into booster_deploy; the policy becomes `models/laai_<phase>.pt` |
| `setup_robot.sh <robot-ip> <phase> [policy.pt]` | Runs `install.sh`, copies booster_deploy to the robot and installs its Python packages there |
| `dry_run.py` | Runs any task's policy on live robot sensors without sending motor commands, then prints a check report |

Edit task files in `phases/<phase>/deploy/`, never the copies inside `booster_deploy`.
