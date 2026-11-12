"""Booster K1: model, booster_deploy settings and helpers shared by every phase.

The values below are copied from booster_deploy/tasks/locomotion/robots/k1/__init__.py
(Booster's `k1_walk` task). booster_deploy sends the same PD gains to the real robot and
builds the policy's observations in the same joint order, so a policy trained with them
runs on the K1 unchanged. Moved here from phases/walk/task.py (run 2) without changes, so
the sit and push phases use the same robot as the walk.
"""

import mujoco

from k1bike.paths import k1_xml

# --- Values copied from booster_deploy (k1_walk) ------------------------------------------------
# booster_deploy joint order (all 22 joints).
DEPLOY_JOINTS = [
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
DEFAULT_POS = [
    0,
    0,
    0.2,
    -1.25,
    0,
    -0.5,
    0.2,
    1.25,
    0,
    0.5,
    -0.15,
    0,
    0,
    0.3,
    -0.15,
    0,
    -0.15,
    0,
    0,
    0.3,
    -0.15,
    0,
]
STIFFNESS = [
    4.0,
    4.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    20.0,
    100.0,
    100.0,
    100.0,
    100.0,
    65.0,
    65.0,
    100.0,
    100.0,
    100.0,
    100.0,
    65.0,
    65.0,
]
DAMPING = [
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
]
EFFORT = [
    6.0,
    6.0,
    14.0,
    14.0,
    14.0,
    14.0,
    14.0,
    14.0,
    14.0,
    14.0,
    30.0,
    20.0,
    15.0,
    35.0,
    24.0,
    15.0,
    30.0,
    20.0,
    15.0,
    35.0,
    24.0,
    15.0,
]
# The 20 joints the policy controls, in booster_deploy's policy order.
POLICY_JOINTS = [
    "aaleft_shoulder_pitch_joint",
    "aaright_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
ACTION_SCALE = 0.25
JOINT_VEL_SCALE = 0.1
HISTORY = 10
BASE_HEIGHT = 0.53
# Motor command delay, in physics steps of 0.005 s: 0..4 = 0..20 ms.
DELAY_MAX_STEPS = 4
# ------------------------------------------------------------------------------------------------

POLICY_JOINT_REGEX = tuple(f"^{n}$" for n in POLICY_JOINTS)
# Bodies with real mass/inertia (camera frames and tiny ankle cross links excluded).
MASS_BODIES = (
    r"^trunk$",
    r"^aahead_.*_link$",
    r"^aa(left|right)_shoulder_pitch_link$",
    r"^(left|right)_(shoulder_roll|elbow_pitch|elbow_yaw)_link$",
    r"^(left|right)_(hip_pitch|hip_roll|hip_yaw|knee_pitch|ankle_roll)_link$",
)
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
FOOT_SITES = ("left_foot", "right_foot")


def k1_spec() -> mujoco.MjSpec:
    """Booster's K1 model, adapted for mjlab: no ground/actuators/old sensors, named foot
    geoms, foot sites, and the IMU sensors mjlab's velocity task expects."""
    spec = mujoco.MjSpec.from_file(str(k1_xml()))
    for geom in list(spec.geoms):
        if geom.name == "ground":
            spec.delete(geom)
    for light in list(spec.lights):
        spec.delete(light)
    for act in list(spec.actuators):
        spec.delete(act)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for side in ("left", "right"):
        foot = spec.body(f"{side}_ankle_roll_link")
        for geom in foot.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                geom.name = f"{side}_foot_collision"
                geom.friction[0] = 0.8
        foot.add_site(name=f"{side}_foot", pos=[0.026, 0.0, -0.038], size=[0.01, 0.01, 0.01])
    spec.add_sensor(
        name="imu_ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu",
    )
    spec.add_sensor(
        name="imu_lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu",
    )
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="trunk",
    )
    return spec


def name_collision_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Give every unnamed collision geom the name `<body>_collision<i>` (feet keep theirs).

    Booster's model leaves them unnamed, so mjlab collision rules and contact sensors can't
    select them by regex. Only names change; the physics is the same.
    """
    for body in spec.bodies:
        colliding = [g for g in body.geoms if g.contype or g.conaffinity]
        for i, geom in enumerate(colliding):
            if not geom.name:
                geom.name = f"{body.name}_collision{i}"
    return spec


def hold_pose_torques(model: mujoco.MjModel, data: mujoco.MjData, target) -> None:
    """Set `data.ctrl` so the K1's torque motors (Booster's raw K1_22dof.xml) hold the
    joint angles in `target` (a full qpos) with the k1_walk PD gains and effort limits.

    For viewing and tests of scenes built from the raw model, not for training
    (mjlab adds its own position actuators).
    """
    gains = {
        name: (kp, kd, effort)
        for name, kp, kd, effort in zip(DEPLOY_JOINTS, STIFFNESS, DAMPING, EFFORT, strict=True)
    }
    for a in range(model.nu):
        joint = model.actuator_trnid[a, 0]
        kp, kd, effort = gains[model.joint(joint).name.rsplit("/", 1)[-1]]
        qadr, dadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        torque = kp * (target[qadr] - data.qpos[qadr]) - kd * data.qvel[dadr]
        data.ctrl[a] = min(max(torque, -effort), effort)
