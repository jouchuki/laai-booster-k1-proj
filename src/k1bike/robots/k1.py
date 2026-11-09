"""Booster K1 model and constants, prepared for mjlab.

Booster's K1_22dof.xml is written for their own simulator setup: it has its own floor,
torque motors and differently named sensors. mjlab wants a bare robot that it adds
terrain and position actuators to, so `k1_spec()` edits the model on load.

The PD gains, effort limits and default pose are copied from Booster's `k1_walk` deploy
config (booster_deploy/tasks/locomotion/robots/k1/__init__.py). booster_deploy
sends those same gains to the real robot, so training with them keeps sim and robot
consistent. This module only needs mujoco, so it is tested on any machine.
"""

import mujoco

from k1bike.paths import k1_xml

# Joints the walking policy controls. The head is held still, as in Booster's k1_walk.
HEAD_JOINTS = ("aahead_yaw_joint", "aahead_pitch_joint")

# Default standing pose (rad), from k1_walk. Joints not listed are 0.
DEFAULT_JOINT_POS = {
    "aaleft_shoulder_pitch_joint": 0.2,
    "aaright_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": -1.25,
    "right_shoulder_roll_joint": 1.25,
    "left_elbow_yaw_joint": -0.5,
    "right_elbow_yaw_joint": 0.5,
    ".*_hip_pitch_joint": -0.15,
    ".*_knee_pitch_joint": 0.3,
    ".*_ankle_pitch_joint": -0.15,
}

# PD gains and effort limits (Nm), grouped by joint, from k1_walk.
# (stiffness, damping, effort_limit)
ACTUATOR_GROUPS = {
    "head": ((r"aahead_.*_joint",), 4.0, 1.0, 6.0),
    "arm": (
        (r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_elbow_.*_joint"),
        20.0,
        2.0,
        14.0,
    ),
    "hip_pitch": ((r".*_hip_pitch_joint",), 100.0, 2.0, 30.0),
    "hip_roll": ((r".*_hip_roll_joint",), 100.0, 2.0, 20.0),
    "hip_yaw": ((r".*_hip_yaw_joint",), 100.0, 2.0, 15.0),
    "knee": ((r".*_knee_pitch_joint",), 100.0, 2.0, 35.0),
    "ankle_pitch": ((r".*_ankle_pitch_joint",), 65.0, 1.0, 24.0),
    "ankle_roll": ((r".*_ankle_roll_joint",), 65.0, 1.0, 15.0),
}

FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
BASE_BODY = "trunk"
IMU_SITE = "imu"

# Trunk height when standing in DEFAULT_JOINT_POS with feet on the ground (m).
# Checked by tests/test_k1_robot.py.
STANDING_HEIGHT = 0.55


def k1_spec() -> mujoco.MjSpec:
    """K1 model edited for mjlab: no floor, no actuators, named collision geoms,
    foot sites, and the IMU sensors mjlab's velocity task reads."""
    spec = mujoco.MjSpec.from_file(str(k1_xml()))

    # mjlab adds its own terrain and position actuators.
    for geom in list(spec.worldbody.geoms):
        if geom.name == "ground":
            spec.delete(geom)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for sensor in list(spec.sensors):
        spec.delete(sensor)

    # Name collision geoms (the non-mesh ones) so configs can select them by regex.
    for body in spec.bodies:
        collision = [
            g for g in body.geoms if g.contype != 0 and g.type != mujoco.mjtGeom.mjGEOM_MESH
        ]
        for i, geom in enumerate(collision):
            if body.name in ("left_ankle_roll_link", "right_ankle_roll_link"):
                side = body.name.split("_")[0]
                geom.name = f"{side}_foot_collision"
            else:
                geom.name = f"{body.name}_collision{i}"

    # Sites under the soles, used for foot height, clearance and slip rewards.
    for side in ("left", "right"):
        spec.body(f"{side}_ankle_roll_link").add_site(name=f"{side}_foot", pos=[0.026, 0, -0.038])

    spec.add_sensor(
        name="imu_ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=IMU_SITE,
    )
    spec.add_sensor(
        name="imu_lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=IMU_SITE,
    )
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname=BASE_BODY,
    )
    return spec
