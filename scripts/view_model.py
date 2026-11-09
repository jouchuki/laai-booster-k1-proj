"""Open a scene in the MuJoCo viewer.

Linux host (from the repository root):
    uv run python scripts/view_model.py bike
    uv run python scripts/view_model.py bike --push 3
    uv run python scripts/view_model.py k1_bike

macOS host (native viewer requires mjpython):
    uv run mjpython scripts/view_model.py bike           # bike alone
    uv run mjpython scripts/view_model.py bike --push 3  # bike pushed at 3 m/s
    uv run mjpython scripts/view_model.py k1_bike        # K1 next to the bike

See docs/commands.md for setup and platform-specific troubleshooting.
"""

import argparse
import time

import mujoco
import mujoco.viewer

from k1bike.scene import bike_scene, k1_bike_scene

SCENES = {"bike": bike_scene, "k1_bike": k1_bike_scene}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--push", type=float, default=0.0, help="initial bike speed (m/s)")
    args = parser.parse_args()

    model = SCENES[args.scene]().compile()
    data = mujoco.MjData(model)
    if args.push:
        bike_root = model.joint("root" if args.scene == "bike" else "bike/root")
        data.qvel[model.jnt_dofadr[bike_root.id]] = args.push

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
