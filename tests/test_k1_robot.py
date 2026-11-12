"""Checks on the shared K1 module (src/k1bike/robots/k1.py). Needs only mujoco."""

import mujoco
import numpy as np

from k1bike.paths import k1_xml
from k1bike.robots.k1 import (
    BASE_HEIGHT,
    DAMPING,
    DEFAULT_POS,
    DEPLOY_JOINTS,
    EFFORT,
    POLICY_JOINTS,
    STIFFNESS,
    hold_pose_torques,
    k1_spec,
    name_collision_geoms,
)


def test_settings_cover_every_joint_in_the_model():
    model = mujoco.MjModel.from_xml_path(str(k1_xml()))
    hinges = {model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != 0}
    assert set(DEPLOY_JOINTS) == hinges
    assert len(DEFAULT_POS) == len(STIFFNESS) == len(DAMPING) == len(EFFORT) == len(DEPLOY_JOINTS)
    assert set(POLICY_JOINTS) == hinges - {"aahead_yaw_joint", "aahead_pitch_joint"}


def test_naming_collision_geoms_changes_only_names():
    plain = k1_spec().compile()
    named = name_collision_geoms(k1_spec()).compile()
    for field in ("geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_size"):
        np.testing.assert_array_equal(getattr(plain, field), getattr(named, field))
    colliding = [
        g for g in range(named.ngeom) if named.geom_contype[g] or named.geom_conaffinity[g]
    ]
    assert len(colliding) == 20
    assert all("_collision" in named.geom(g).name for g in colliding)


def test_k1_stands_in_its_default_pose_with_the_pd_gains():
    """Booster's raw model (torque motors) held at DEFAULT_POS stands near BASE_HEIGHT."""
    spec = mujoco.MjSpec.from_file(str(k1_xml()))
    model = spec.compile()
    data = mujoco.MjData(model)
    target = model.qpos0.copy()
    for name, angle in zip(DEPLOY_JOINTS, DEFAULT_POS, strict=True):
        target[model.jnt_qposadr[model.joint(name).id]] = angle
    data.qpos[:] = target
    data.qpos[2] = BASE_HEIGHT + 0.02
    for _ in range(int(2.0 / model.opt.timestep)):
        hold_pose_torques(model, data, target)
        mujoco.mj_step(model, data)
    trunk = model.body("trunk").id
    assert abs(data.xpos[trunk, 2] - BASE_HEIGHT) < 0.03
    assert data.xmat[trunk].reshape(3, 3)[2, 2] > 0.98, "upright"
