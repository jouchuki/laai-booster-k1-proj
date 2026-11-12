# ROS 2 Docker experiment (legacy, unvalidated)

An early optional experiment to run Booster's ROS 2 bridge in an Ubuntu 22.04 container. It is
not part of the current workflow: simulation, training and robot deployment use mjlab and
booster_deploy (see [docs/commands.md](../../docs/commands.md)).

| File | What |
|---|---|
| `install_ros2_humble.sh` | Runs **inside** the Ubuntu 22.04 container: installs ROS 2 Humble, builds `booster_robotics_sdk_ros2`, launches the K1 bridge. The launch part has not been validated. Set `BOOSTER_IP` to the robot's address first. |
| `rebuild_container.sh` | On the Linux or macOS host with Docker: rebuilds and restarts the `ros2-booster-k1` container from a local `Dockerfile` (gitignored) in the current folder. |

Local ROS workspaces (`ros*/`) and the `Dockerfile` are gitignored.
