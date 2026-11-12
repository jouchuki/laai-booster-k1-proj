# Robot: tools for the real K1

Booster's `booster_deploy` (cloned by `scripts/setup_booster.sh`, gitignored) runs our
policies in MuJoCo and on the robot. Each experiment keeps its own booster_deploy task files in
`experiments/<experiment>/deploy/`; the tools here install and test them for any experiment.

Commands for every step (install, Booster's MuJoCo, copy to the robot, dry run, robot test,
controls, logs) are in [docs/commands.md](../docs/commands.md).

| File | What |
|---|---|
| `install.sh <experiment> [policy.pt]` | Copies `experiments/<experiment>/deploy/*.py`, `dry_run.py`, the seated poses (`models/seated_poses/`) and a policy (default: the newest run's `policy.pt`; none for an experiment without trained runs, such as `sit`) into booster_deploy; the policy becomes `models/laai_<experiment>.pt` |
| `setup_robot.sh <robot-ip> <experiment> [policy.pt]` | Runs `install.sh`, copies booster_deploy to the robot and installs its Python packages there |
| `dry_run.py` | Runs any task's policy on live robot sensors without sending motor commands, then prints a check report |
| `robot_info.py` | Connects with Booster's `boosteros` SDK and prints the robot's model, mode and joint count (Linux, on the robot's network) |

Edit task files in `experiments/<experiment>/deploy/`, never the copies inside `booster_deploy`.
Robot logins (user, IP, password) are never committed; see [docs/handover/](../docs/handover/README.md)
for the local credentials file.
