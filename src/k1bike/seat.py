"""Put the K1 on the bike: hands clipped to the grips, pelvis resting on the saddle.

Two different kinds of connection, matching the physical setup:

- Hands: the orange plastic clips hold each hand rigidly to the handlebar
  (docs/bike-dimensions.md, "Passive hand clips"). Each clip is a MuJoCo *weld*
  equality constraint between a site on the robot's forearm and a site on the
  bike's fork/handlebar body. A weld removes all six relative degrees of freedom
  (3 translation + 3 rotation), so the hands steer the bike and the bike's weight
  hangs from the hands. The steering joint itself stays free.
- Seat: nothing. The robot sits on the saddle only through ordinary contact
  between its pelvis geoms and the saddle box, with gravity holding it down.
  It can push itself up, slide, and lift off, which is what lets it learn to
  walk the bike with its feet while it is rewarded for staying seated
  (see `seat_contact`).

The seated starting pose comes from `scripts/solve_seated_pose.py`, which solves
for a pose with the pelvis on the saddle, feet flat on the floor and both clip
points on the grips. It is stored in `k1_seated_pose.json` next to this module.

Clip geometry (an assumption until the clip's frames are measured): the clip is
collinear with the forearm. The robot's hand occupies clip positions s = 4-8 cm,
with its centre at s = 6 cm, 2 cm in from the forearm tip. The handlebar is held
at s = 2 cm, 4 cm further along the same axis. So the handlebar centre sits 2 cm
beyond the K1's forearm tip, along the forearm.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SIDES = ("left", "right")

# Robot side. The K1 has no hands: the forearm (elbow yaw link) ends in a rounded
# tip at +-0.228 m along the link's y axis (measured from the collision mesh).
HAND_LINKS = {"left": "left_elbow_yaw_link", "right": "right_elbow_yaw_link"}
FOREARM_TIP = 0.228
CLIP_HAND_CENTRE_INSET = 0.02  # hand centre (s = 6 cm) is 2 cm in from the tip (s = 8 cm)
CLIP_HAND_TO_BAR = 0.04  # bar centre (s = 2 cm) is 4 cm beyond the hand centre
HAND_CLIP_OFFSET = FOREARM_TIP - CLIP_HAND_CENTRE_INSET + CLIP_HAND_TO_BAR  # 0.248 m

# Bodies that may rest on the saddle. The trunk's bottom cylinder is the K1's pelvis.
SEAT_BODY = "trunk"

# Bike side (names inside the bike model, before any prefix).
FORK_BODY = "fork"
SADDLE_GEOM = "saddle"
GRIP_SITES = {"left": "left_grip", "right": "right_grip"}

SEATED_POSE_FILE = Path(__file__).with_name("k1_seated_pose.json")
SEATED_KEY = "seated"


@dataclass(frozen=True)
class SeatedPose:
    """K1 pose relative to the bike frame's nominal pose.

    trunk_pos / trunk_quat: trunk body pose in the bike frame body's coordinates.
    joints: hinge angles (rad) by K1 joint name, without prefix.
    """

    trunk_pos: tuple[float, float, float]
    trunk_quat: tuple[float, float, float, float]
    joints: dict[str, float]


def load_seated_pose(path: Path = SEATED_POSE_FILE) -> SeatedPose:
    raw = json.loads(path.read_text())
    return SeatedPose(
        trunk_pos=tuple(raw["trunk_pos"]),
        trunk_quat=tuple(raw["trunk_quat"]),
        joints=dict(raw["joints"]),
    )


def add_hand_clip_sites(spec: mujoco.MjSpec, prefix: str = "") -> None:
    """Add `<side>_hand_clip` sites on the K1's forearms: where the clip holds the bar."""
    for side in SIDES:
        sign = 1.0 if side == "left" else -1.0
        spec.body(f"{prefix}{HAND_LINKS[side]}").add_site(
            name=f"{prefix}{side}_hand_clip",
            pos=[0.0, sign * HAND_CLIP_OFFSET, 0.0],
            size=[0.008, 0, 0],
            rgba=[1.0, 0.5, 0.0, 1.0],
            group=4,
        )


def seated_qpos(
    model: mujoco.MjModel,
    pose: SeatedPose,
    robot_prefix: str = "",
    bike_prefix: str = "bike/",
    robot_root: str = "world_joint",
) -> np.ndarray:
    """Full qpos with the bike at its nominal pose and the K1 seated on it."""
    qpos = model.qpos0.copy()
    bike = int(model.jnt_qposadr[model.joint(f"{bike_prefix}root").id])
    bike_pos, bike_quat = qpos[bike : bike + 3], qpos[bike + 3 : bike + 7]

    root = int(model.jnt_qposadr[model.joint(f"{robot_prefix}{robot_root}").id])
    offset = np.zeros(3)
    mujoco.mju_rotVecQuat(offset, np.asarray(pose.trunk_pos, dtype=float), bike_quat)
    qpos[root : root + 3] = bike_pos + offset
    mujoco.mju_mulQuat(qpos[root + 3 : root + 7], bike_quat, np.asarray(pose.trunk_quat, float))

    for name, angle in pose.joints.items():
        qpos[model.jnt_qposadr[model.joint(f"{robot_prefix}{name}").id]] = angle
    return qpos


def clip_hands_to_bike(
    spec: mujoco.MjSpec,
    pose: SeatedPose | None = None,
    robot_prefix: str = "",
    bike_prefix: str = "bike/",
    robot_root: str = "world_joint",
) -> np.ndarray:
    """Weld the K1's hand clip sites to the handlebar and add a "seated" keyframe.

    `spec` must already contain the K1 (with `add_hand_clip_sites`) and the bike.
    Prefixes let this run on a combined scene such as mjlab's (`robot/`, `bike/`);
    it can be passed as mjlab's scene `spec_fn`.

    The bike-side site of each weld is placed exactly where the robot's clip site is
    in the seated pose, with the same orientation. The welds therefore start with zero
    error and hold the clip at that pose relative to the handlebar. Contacts between
    each forearm and the fork are excluded: the clipped forearm overlaps the bar's
    collision capsule, and the weld already defines how the two interact.

    Returns the seated qpos (also stored as keyframe "seated").
    """
    pose = pose or load_seated_pose()
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos = seated_qpos(model, pose, robot_prefix, bike_prefix, robot_root)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)

    fork = model.body(f"{bike_prefix}{FORK_BODY}").id
    fork_pos, fork_mat = data.xpos[fork], data.xmat[fork].reshape(3, 3)
    for side in SIDES:
        clip = model.site(f"{robot_prefix}{side}_hand_clip").id
        local_pos = fork_mat.T @ (data.site_xpos[clip] - fork_pos)
        local_quat = np.zeros(4)
        mujoco.mju_mat2Quat(local_quat, (fork_mat.T @ data.site_xmat[clip].reshape(3, 3)).ravel())
        grip_site = f"{bike_prefix}{side}_grip_clip"
        spec.body(f"{bike_prefix}{FORK_BODY}").add_site(
            name=grip_site,
            pos=local_pos.tolist(),
            quat=local_quat.tolist(),
            size=[0.006, 0, 0],
            rgba=[0.1, 0.6, 1.0, 1.0],
            group=4,
        )
        spec.add_equality(
            name=f"{side}_hand_clip",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{robot_prefix}{side}_hand_clip",
            name2=grip_site,
        )
        spec.add_exclude(
            bodyname1=f"{robot_prefix}{HAND_LINKS[side]}",
            bodyname2=f"{bike_prefix}{FORK_BODY}",
        )

    spec.add_key(name=SEATED_KEY, qpos=qpos.tolist())
    return qpos


def seat_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_prefix: str = "",
    bike_prefix: str = "bike/",
) -> float:
    """Normal force (N) of the K1's pelvis on the saddle; 0 when it is not sitting.

    For rewards and terminations: "stay seated" means this stays positive. The K1
    weighs about 19.7 kg (193 N); sitting with feet down puts part of that on the saddle.
    """
    saddle = model.geom(f"{bike_prefix}{SADDLE_GEOM}").id
    seat_body = model.body(f"{robot_prefix}{SEAT_BODY}").id
    force = np.zeros(6)
    total = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        geoms = (contact.geom1, contact.geom2)
        if saddle not in geoms:
            continue
        other = geoms[1] if geoms[0] == saddle else geoms[0]
        if model.geom_bodyid[other] != seat_body:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        total += float(force[0])
    return total
