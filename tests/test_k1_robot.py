"""Checks on the K1 model as prepared for mjlab (k1bike.robots.k1). Needs only mujoco."""

import re

import mujoco
import numpy as np

from k1bike.robots.k1 import (
    ACTUATOR_GROUPS,
    DEFAULT_JOINT_POS,
    FOOT_GEOMS,
    FOOT_SITES,
    STANDING_HEIGHT,
    k1_spec,
)


def _joint_names(model):
    return [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0]


def _standing_robot_with_pd():
    """K1 with Booster's k1_walk PD gains as position actuators, in the default pose."""
    spec = k1_spec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    joints = _joint_names(spec.compile())
    targets = {}
    for patterns, kp, kd, effort in ACTUATOR_GROUPS.values():
        for joint in joints:
            if any(re.fullmatch(p, joint) for p in patterns):
                act = spec.add_actuator(name=joint, target=joint)
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.set_to_position(kp=kp, kv=kd)
                act.forcerange = [-effort, effort]
                act.forcelimited = True
                targets[joint] = next(
                    (v for p, v in DEFAULT_JOINT_POS.items() if re.fullmatch(p, joint)), 0.0
                )
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[2] = STANDING_HEIGHT
    for joint, value in targets.items():
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
        data.ctrl[model.actuator(joint).id] = value
    return model, data


def test_every_joint_has_gains():
    joints = _joint_names(k1_spec().compile())
    assert len(joints) == 22
    patterns = [p for group in ACTUATOR_GROUPS.values() for p in group[0]]
    missing = [j for j in joints if not any(re.fullmatch(p, j) for p in patterns)]
    assert not missing, f"joints without PD gains: {missing}"


def test_default_pose_names_exist():
    joints = _joint_names(k1_spec().compile())
    for pattern in DEFAULT_JOINT_POS:
        assert any(re.fullmatch(pattern, j) for j in joints), pattern


def test_mjlab_needs_are_present():
    model = k1_spec().compile()
    assert model.nu == 0, "mjlab adds its own actuators"
    for name in ("imu_ang_vel", "imu_lin_vel", "root_angmom"):
        model.sensor(name)
    for name in FOOT_SITES:
        model.site(name)
    for name in FOOT_GEOMS:
        model.geom(name)
    names = [model.geom(i).name for i in range(model.ngeom)]
    assert "ground" not in names, "mjlab adds its own terrain"


def test_k1_stands_with_walk_gains():
    """With Booster's gains and default pose, the K1 should stand for 2 s without a policy."""
    model, data = _standing_robot_with_pd()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.qpos[2] > STANDING_HEIGHT - 0.05, f"trunk dropped to {data.qpos[2]:.3f} m"
