"""Open a scene in the MuJoCo viewer.

Linux host (from the repository root):
    uv run python scripts/view_model.py bike
    uv run python scripts/view_model.py bike --paused
    uv run python scripts/view_model.py bike --push 3
    uv run python scripts/view_model.py bike --lean 5 --drop 0.03
    uv run python scripts/view_model.py bike --drop 0.3
    uv run python scripts/view_model.py k1_bike
    uv run python scripts/view_model.py k1_seated

macOS host (native viewer requires mjpython):
    uv run mjpython scripts/view_model.py bike           # bike alone
    uv run mjpython scripts/view_model.py bike --paused
    uv run mjpython scripts/view_model.py bike --push 3  # bike pushed at 3 m/s
    uv run mjpython scripts/view_model.py bike --lean 5 --drop 0.03
    uv run mjpython scripts/view_model.py bike --drop 0.3
    uv run mjpython scripts/view_model.py k1_bike        # K1 next to the bike
    uv run mjpython scripts/view_model.py k1_seated      # K1 seated, hands clipped

See docs/commands.md for setup and platform-specific troubleshooting.
The default bike is models/bike (the scaled USDZ-derived model). Use
--model bike_primitive for the original blue placeholder. k1_seated starts the K1 sitting on the USDZ-derived bike,
hands welded to the grips, and holds that pose with the k1_walk PD gains
(enable site group 4 in the viewer's Rendering panel to see the clip sites).
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from k1bike.paths import BIKE_XML, PRIMITIVE_BIKE_XML
from k1bike.robots.k1 import hold_pose_torques
from k1bike.scene import bike_scene, initialize_bike_state, k1_bike_scene, k1_seated_bike_scene

SCENES = {"bike": bike_scene, "k1_bike": k1_bike_scene, "k1_seated": k1_seated_bike_scene}
# Folder names under models/.
MODELS = {"bike": BIKE_XML, "bike_primitive": PRIMITIVE_BIKE_XML}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument(
        "--model", choices=MODELS, default="bike", help="bike model (models/<name>)"
    )
    parser.add_argument(
        "--paused", action="store_true", help="start paused; Space toggles simulation"
    )
    parser.add_argument("--push", type=float, default=0.0, help="initial bike speed (m/s)")
    parser.add_argument("--lean", type=float, default=0.0, help="initial bike roll angle (degrees)")
    parser.add_argument("--drop", type=float, default=0.0, help="raise the bike before release (m)")
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.push, args.lean, args.drop)):
        parser.error("--push, --lean and --drop must be finite numbers")
    if args.drop < 0:
        parser.error("--drop must be nonnegative")

    bike_xml = MODELS[args.model]
    if not bike_xml.is_file():
        parser.error(f"Bike model is missing: {bike_xml}; see docs/commands.md for conversion")
    if args.scene == "k1_seated" and args.model != "bike":
        parser.error("k1_seated uses models/bike (its seated pose is solved for it)")
    if args.scene == "k1_seated" and (args.push or args.lean or args.drop):
        parser.error("--push, --lean and --drop would move the bike out from under the K1")
    model = SCENES[args.scene](bike_xml=bike_xml).compile()
    data = mujoco.MjData(model)
    hold = None
    if args.scene == "k1_seated":
        mujoco.mj_resetDataKeyframe(model, data, model.key("seated").id)
        hold = data.qpos.copy()
        mujoco.mj_forward(model, data)
    else:
        initialize_bike_state(
            model,
            data,
            prefix="" if args.scene == "bike" else "bike/",
            forward_speed=args.push,
            lean_degrees=args.lean,
            drop_height=args.drop,
        )
    print(f"Model: {args.model} ({bike_xml})", flush=True)
    print(f"Gravity: {model.opt.gravity.tolist()} m/s²", flush=True)
    print(f"Bike: lean={args.lean:g}°, drop={args.drop:g} m, speed={args.push:g} m/s")

    paused = args.paused

    def key_callback(keycode):
        nonlocal paused
        if keycode == ord(" "):
            paused = not paused

    print("Space: pause/resume simulation", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            start = time.time()
            if not paused:
                if hold is not None:
                    hold_pose_torques(model, data, hold)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
