"""Solve for the K1's seated pose on a bike and save it.

Linux and macOS, from the repository root (headless, no viewer):
    uv run --group sim python scripts/solve_seated_pose.py                  # the real bike
    uv run --group sim python scripts/solve_seated_pose.py --bike yellow_bike      # another bike

Writes models/k1_on_bike/<bike>.json, which k1_seated_bike_scene loads for that bike.
Rerun it after changing that bike model, the clip geometry in k1bike/seat.py, or the
targets below. It refuses to write a pose that does not fit (see TOLERANCES); the
report says which target failed, for example feet that cannot reach the floor.

The pose is a least-squares fit, with joints kept inside 90% of their limits:
- pelvis (trunk bottom cylinder) touching the saddle top, centred over it;
- both feet flat on the floor, a little wider than the hips and a little in front of the
  saddle, so part of the weight rests on the saddle (feet behind the hips is a crouch);
- each hand clip site on its handlebar grip, with the forearm horizontal
  (the owner reports hand and bar centres at about the same height);
- trunk leaning forward about 20 degrees;
- weak pull towards a neutral posture so the redundant joints are well defined.
"""

import argparse
import json
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

from k1bike.paths import BIKES, DEFAULT_BIKE, seated_pose_json
from k1bike.robots.k1 import POLICY_JOINTS
from k1bike.scene import k1_bike_scene
from k1bike.seat import (
    FORK_BODY,
    GRIP_SITES,
    HAND_LINKS,
    SADDLE_GEOM,
    SEAT_BODY,
    SIDES,
    add_hand_clip_sites,
)

BIKE = "bike/"
FOOT_WIDTH = 0.13  # target |y| of each foot centre (hips are at +-0.096)
FOOT_FORWARD = 0.06  # target x of each foot centre in front of the saddle's seat point
# Forward trunk lean, as a rider leans on a balance bike. Sitting upright puts the K1's mass
# behind its feet; on a bike with the saddle close to the rear axle it then tips the bike
# backwards through the hand clips.
TRUNK_PITCH = 0.35  # rad (20 deg)
SADDLE_PENETRATION = 0.001  # start just in contact, not hovering
# Largest fit errors accepted, in metres: a pose outside these is not written.
TOLERANCES = {"saddle": 0.003, "clip": 0.002, "foot": 0.003, "penetration": 0.005}
# Neutral posture for the weak regularizer: knees and hips bent as when sitting.
NEUTRAL = {
    ".*_hip_pitch_joint": -0.6,
    ".*_knee_pitch_joint": 0.9,
    ".*_ankle_pitch_joint": -0.3,
}


def _model(bike_xml) -> mujoco.MjModel:
    spec = k1_bike_scene(bike_pos=(0.0, 0.0, 0.0), bike_xml=bike_xml)
    add_hand_clip_sites(spec)
    return spec.compile()


def _neutral(name: str) -> float:
    import re

    return next((v for p, v in NEUTRAL.items() if re.fullmatch(p, name)), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bike", choices=BIKES, default=DEFAULT_BIKE, help="models/<bike>")
    parser.add_argument("--force", action="store_true", help="write the pose even if it fails")
    args = parser.parse_args()
    bike_xml = BIKES[args.bike]
    out = seated_pose_json(bike_xml)
    print(f"bike: {args.bike} ({bike_xml})")
    model = _model(bike_xml)
    data = mujoco.MjData(model)
    root = int(model.jnt_qposadr[model.joint("world_joint").id])
    joints = [
        model.joint(j).name
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and not model.joint(j).name.startswith(BIKE)
        and model.joint(j).name in POLICY_JOINTS  # the head is not posed
    ]
    adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joints])
    lo = np.array([0.9 * model.joint(n).range[0] for n in joints])
    hi = np.array([0.9 * model.joint(n).range[1] for n in joints])
    neutral = np.array([_neutral(n) for n in joints])

    saddle = model.geom(f"{BIKE}{SADDLE_GEOM}").id
    seat_body = model.body(SEAT_BODY).id
    pelvis = next(  # the trunk's bottom cylinder
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == seat_body
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
    )
    floor = model.geom("floor").id
    feet = {
        side: next(
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.body(f"{side}_ankle_roll_link").id
            and model.geom_contype[g]
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        )
        for side in SIDES
    }
    seat_site = model.site(f"{BIKE}seat").id
    fromto = np.zeros(6)

    def apply(x: np.ndarray) -> None:
        data.qpos[:] = model.qpos0
        tx, tz, pitch = x[:3]
        data.qpos[root : root + 3] = [tx, 0.0, tz]
        mujoco.mju_axisAngle2Quat(data.qpos[root + 3 : root + 7], np.array([0, 1.0, 0]), pitch)
        data.qpos[adr] = x[3:]
        mujoco.mj_kinematics(model, data)

    def residuals(x: np.ndarray) -> np.ndarray:
        apply(x)
        r = []
        dist = mujoco.mj_geomDistance(model, data, pelvis, saddle, 0.5, fromto)
        r += [200 * (dist + SADDLE_PENETRATION)]
        r += [5 * (data.geom_xpos[pelvis, 0] - data.site_xpos[seat_site, 0])]
        for side, sign in (("left", 1.0), ("right", -1.0)):
            foot = feet[side]
            r += [200 * mujoco.mj_geomDistance(model, data, foot, floor, 0.5, fromto)]
            zaxis = data.geom_xmat[foot].reshape(3, 3)[:, 2]
            r += [5 * zaxis[0], 5 * zaxis[1]]
            r += [2 * (data.geom_xpos[foot, 1] - sign * FOOT_WIDTH)]
            r += [2 * (data.geom_xpos[foot, 0] - data.site_xpos[seat_site, 0] - FOOT_FORWARD)]
            clip = model.site(f"{side}_hand_clip").id
            grip = model.site(f"{BIKE}{GRIP_SITES[side]}").id
            r += list(200 * (data.site_xpos[clip] - data.site_xpos[grip]))
            forearm_axis = data.xmat[model.body(HAND_LINKS[side]).id].reshape(3, 3)[:, 1]
            r += [3 * forearm_axis[2]]
        # Left/right symmetry of the legs (mirror: pitch-like joints equal, roll/yaw opposite).
        for name in joints:
            if name.startswith("left_") and ("hip" in name or "knee" in name or "ankle" in name):
                i, j = joints.index(name), joints.index(name.replace("left_", "right_"))
                same = "pitch" in name
                r += [2 * (x[3 + i] - x[3 + j]) if same else 2 * (x[3 + i] + x[3 + j])]
        r += [0.5 * (x[2] - TRUNK_PITCH)]  # trunk leaning forward
        r += list(0.05 * (x[3:] - neutral))
        return np.array(r)

    mujoco.mj_kinematics(model, data)
    seat_x = float(data.site_xpos[seat_site, 0])
    seat_z = float(data.site_xpos[seat_site, 2])
    # Trunk origin starts about 11 cm above the saddle top, as on the first solved bike.
    x0 = np.concatenate([[seat_x, seat_z + 0.11, 0.0], np.clip(neutral, lo, hi)])
    # Arms start reaching forward and down rather than out to the sides.
    for name, value in {
        "aaleft_shoulder_pitch_joint": -0.8,
        "aaright_shoulder_pitch_joint": -0.8,
        "left_shoulder_roll_joint": -1.3,
        "right_shoulder_roll_joint": 1.3,
    }.items():
        x0[3 + joints.index(name)] = value
    bounds = (
        np.concatenate([[-0.5, 0.3, -0.5], lo]),
        np.concatenate([[1.0, 0.8, 0.5], hi]),
    )
    result = least_squares(residuals, x0, bounds=bounds, xtol=1e-12, ftol=1e-12, max_nfev=5000)
    x = result.x
    apply(x)

    # Report fit quality against the tolerances.
    print(f"solver: {result.message} (cost {result.cost:.2e})")
    saddle_gap = mujoco.mj_geomDistance(model, data, pelvis, saddle, 0.5, fromto)
    fit = {"pelvis_saddle_mm": saddle_gap * 1000}
    failures = []
    if abs(saddle_gap + SADDLE_PENETRATION) > TOLERANCES["saddle"]:
        failures.append("pelvis does not rest on the saddle")
    for side in SIDES:
        clip = model.site(f"{side}_hand_clip").id
        grip = model.site(f"{BIKE}{GRIP_SITES[side]}").id
        err = float(np.linalg.norm(data.site_xpos[clip] - data.site_xpos[grip]))
        foot_gap = mujoco.mj_geomDistance(model, data, feet[side], floor, 0.5, fromto)
        fit[f"{side}_clip_to_grip_mm"] = err * 1000
        fit[f"{side}_foot_to_floor_mm"] = foot_gap * 1000
        if err > TOLERANCES["clip"]:
            failures.append(f"{side} hand clip does not reach its grip")
        if abs(foot_gap) > TOLERANCES["foot"]:
            failures.append(f"{side} foot is not flat on the floor")
    mujoco.mj_forward(model, data)
    worst = min((data.contact[i].dist for i in range(data.ncon)), default=0.0)
    fit["deepest_penetration_mm"] = -worst * 1000
    if -worst > TOLERANCES["penetration"]:
        failures.append("robot and bike overlap")
    for name, value in fit.items():
        print(f"  {name}: {value:.2f}")
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -0.003:
            names = [model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
            print(f"  overlap {names[0]} / {names[1]}: {-c.dist * 1000:.1f} mm")
    trunk_pitch = float(np.degrees(x[2]))
    print(f"  trunk pitch: {trunk_pitch:.1f} deg")
    if failures:
        print("FIT FAILED: " + "; ".join(failures))
        if not args.force:
            sys.exit(f"{out} not written (use --force to write it anyway)")

    # Trunk pose relative to the bike frame body (the bike sits at the world origin).
    frame = model.body(f"{BIKE}frame").id
    frame_pos, frame_mat = data.xpos[frame], data.xmat[frame].reshape(3, 3)
    trunk_pos = frame_mat.T @ (data.qpos[root : root + 3] - frame_pos)
    frame_quat_inv = np.zeros(4)
    mujoco.mju_negQuat(frame_quat_inv, data.xquat[frame])
    trunk_quat = np.zeros(4)
    mujoco.mju_mulQuat(trunk_quat, frame_quat_inv, data.qpos[root + 3 : root + 7])
    pose = {
        "description": (
            f"K1 seated on models/{args.bike}; generated by scripts/solve_seated_pose.py "
            f"--bike {args.bike}. trunk_pos/trunk_quat are relative to the bike frame body; "
            "joints in rad."
        ),
        "bike": args.bike,
        "fork_body": FORK_BODY,
        "fit": {k: round(v, 3) for k, v in fit.items()}
        | {"trunk_pitch_deg": round(trunk_pitch, 1)},
        "trunk_pos": trunk_pos.round(6).tolist(),
        "trunk_quat": trunk_quat.round(6).tolist(),
        "joints": {n: round(float(v), 6) for n, v in zip(joints, x[3:], strict=True)},
    }
    out.write_text(json.dumps(pose, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
