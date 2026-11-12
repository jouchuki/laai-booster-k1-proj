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
The default bike is the scaled USDZ-derived model. Use --model primitive for the
original blue prototype, or --model real_bike for the uploaded textured scan.
k1_seated starts the K1 sitting on the USDZ-derived bike,
hands welded to the grips, and holds that pose with the k1_walk PD gains
(enable site group 4 in the viewer's Rendering panel to see the clip sites).
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer

from k1bike.paths import BIKE_XML, REAL_BIKE_XML, SKETCHFAB_BIKE_XML
from k1bike.robots.k1 import hold_pose_torques
from k1bike.scene import bike_scene, initialize_bike_state, k1_bike_scene, k1_seated_bike_scene

SCENES = {"bike": bike_scene, "k1_bike": k1_bike_scene, "k1_seated": k1_seated_bike_scene}
MODELS = {"sketchfab": SKETCHFAB_BIKE_XML, "primitive": BIKE_XML, "real_bike": REAL_BIKE_XML}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--model", choices=MODELS, default="sketchfab", help="bike asset to load")
    parser.add_argument(
        "--paused", action="store_true", help="start paused; Space toggles simulation"
    )
    parser.add_argument("--push", type=float, default=0.0, help="initial bike speed (m/s)")
    parser.add_argument(
        "--steer", type=float, default=0.0, help="initial steering angle (degrees; then passive)"
    )
    parser.add_argument("--lean", type=float, default=0.0, help="initial bike roll angle (degrees)")
    parser.add_argument("--drop", type=float, default=0.0, help="raise the bike before release (m)")
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.push, args.steer, args.lean, args.drop)):
        parser.error("--push, --steer, --lean and --drop must be finite numbers")
    if args.drop < 0:
        parser.error("--drop must be nonnegative")

    bike_xml = MODELS[args.model]
    if not bike_xml.is_file():
        parser.error(f"Bike model is missing: {bike_xml}; see docs/commands.md for conversion")
    if args.scene == "k1_seated" and args.model != "sketchfab":
        parser.error("k1_seated uses the sketchfab bike (its seated pose is solved for it)")
    if args.scene == "k1_seated" and (args.push or args.steer or args.lean or args.drop):
        parser.error("--push, --steer, --lean and --drop would disrupt the K1's seated pose")
    model = SCENES[args.scene](bike_xml=bike_xml).compile()
    data = mujoco.MjData(model)
    prefix = "" if args.scene == "bike" else "bike/"
    steering = model.joint(f"{prefix}steer")
    angle = math.radians(args.steer)
    if model.jnt_limited[steering.id] and not (
        model.jnt_range[steering.id, 0] <= angle <= model.jnt_range[steering.id, 1]
    ):
        parser.error("--steer must be within the model's steering limits")

    hold = None

    def reset():
        nonlocal hold
        if args.scene == "k1_seated":
            mujoco.mj_resetDataKeyframe(model, data, model.key("seated").id)
            hold = data.qpos.copy()
            mujoco.mj_forward(model, data)
        else:
            mujoco.mj_resetData(model, data)
            data.qpos[model.jnt_qposadr[steering.id]] = angle
            initialize_bike_state(
                model,
                data,
                prefix=prefix,
                forward_speed=args.push,
                lean_degrees=args.lean,
                drop_height=args.drop,
            )

    reset()
    print(f"Model: {args.model} ({bike_xml})", flush=True)
    print(f"Gravity: {model.opt.gravity.tolist()} m/s²", flush=True)
    print(f"Bike: lean={args.lean:g}°, drop={args.drop:g} m, speed={args.push:g} m/s")
    print(f"Mass: {model.body_subtreemass[model.body(f'{prefix}frame').id]:g} kg")
    print(f"Initial steering: {args.steer:g}° (passive after release)")

    paused = args.paused
    reset_requested = False

    def key_callback(keycode):
        nonlocal paused, reset_requested
        if keycode == ord(" "):
            paused = not paused
        elif keycode in (ord("R"), ord("r")):
            reset_requested = True

    print("Space: pause/resume simulation; R: repeat initial push", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body(f"{prefix}frame").id
        while viewer.is_running():
            start = time.time()
            if reset_requested:
                reset()
                reset_requested = False
            if not paused:
                if hold is not None:
                    hold_pose_torques(model, data, hold)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
