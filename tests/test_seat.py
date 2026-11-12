"""K1 seated on the bike: hands welded to the grips, pelvis resting on the saddle by contact."""

import mujoco
import numpy as np
import pytest

from k1bike.paths import PRIMITIVE_BIKE_XML
from k1bike.robots.k1 import hold_pose_torques
from k1bike.scene import k1_seated_bike_scene
from k1bike.seat import FORK_BODY, HAND_LINKS, SADDLE_GEOM, SEAT_BODY, SIDES, seat_contact


@pytest.fixture(scope="module")
def model():
    return k1_seated_bike_scene().compile()


def _seated(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("seated").id)
    mujoco.mj_forward(model, data)
    return data


def _clip_error(model, data):
    return max(
        np.linalg.norm(
            data.site_xpos[model.site(f"{side}_hand_clip").id]
            - data.site_xpos[model.site(f"bike/{side}_grip_clip").id]
        )
        for side in SIDES
    )


def _pelvis_saddle_distance(model, data):
    trunk = model.body(SEAT_BODY).id
    pelvis = next(
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == trunk and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
    )
    return mujoco.mj_geomDistance(
        model, data, pelvis, model.geom(f"bike/{SADDLE_GEOM}").id, 1.0, np.zeros(6)
    )


def _hold_pose(model, data, target, seconds, push_up=0.0):
    """Step with Booster's k1_walk PD gains on the K1's torque motors, holding `target`."""
    trunk = model.body("trunk").id
    for _ in range(int(seconds / model.opt.timestep)):
        hold_pose_torques(model, data, target)
        data.xfrc_applied[trunk, 2] = push_up
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)


def test_hands_are_welded_to_the_handlebar_not_the_seat(model):
    assert model.neq == 2
    fork = model.body("bike/fork").id
    for i in range(model.neq):
        assert model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD
        assert model.eq_objtype[i] == mujoco.mjtObj.mjOBJ_SITE
        robot_site, bike_site = model.eq_obj1id[i], model.eq_obj2id[i]
        assert model.site(robot_site).name.endswith("_hand_clip")
        assert model.site_bodyid[bike_site] == fork, "clips hold the steerable fork"


def test_seated_keyframe_is_consistent(model):
    data = _seated(model)
    welds = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert np.abs(data.efc_pos[welds]).max() < 1e-9, "welds start with zero error"
    assert _clip_error(model, data) < 1e-6
    for side in SIDES:
        clip = data.site_xpos[model.site(f"{side}_hand_clip").id]
        grip = data.site_xpos[model.site(f"bike/{side}_grip").id]
        assert np.linalg.norm(clip - grip) < 1e-3, "clip sits on the bike's grip site"
    assert -0.002 < _pelvis_saddle_distance(model, data) < 0.0005, "pelvis touches the saddle"


def _blocked_bike_bodies(model):
    """Robot-bike collision geom pairs that cannot touch, as (robot body, bike body) names,
    and the number of robot-robot pairs that can."""
    excluded = set()
    for i in range(model.nexclude):
        a, b = divmod(int(model.exclude_signature[i]), 1 << 16)
        excluded.add(frozenset((a, b)))

    def can_touch(a, b):
        ba, bb = model.geom_bodyid[a], model.geom_bodyid[b]
        if ba == bb or model.body_parentid[ba] == bb or model.body_parentid[bb] == ba:
            return False
        if frozenset((ba, bb)) in excluded:
            return False
        return bool(
            model.geom_contype[a] & model.geom_conaffinity[b]
            or model.geom_contype[b] & model.geom_conaffinity[a]
        )

    def name(g):
        return model.body(model.geom_bodyid[g]).name

    colliding = [
        g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    bike = [g for g in colliding if name(g).startswith("bike/")]
    robot = [g for g in colliding if g not in bike and model.geom_bodyid[g] != 0]
    blocked = {(name(r), name(b)) for r in robot for b in bike if not can_touch(r, b)}
    self_pairs = sum(can_touch(a, b) for i, a in enumerate(robot) for b in robot[i + 1 :])
    return blocked, self_pairs, len(bike)


def test_robot_collides_with_itself_and_the_bike(model):
    """No passing through: the K1 keeps all its self-collisions and touches every bike
    collision geom, except each welded forearm and the fork (its hand overlaps the bar)."""
    blocked, self_pairs, bike_geoms = _blocked_bike_bodies(model)
    assert self_pairs == 174, "same self-collision as the walk task"
    assert bike_geoms == 12
    assert blocked == {(HAND_LINKS[side], f"bike/{FORK_BODY}") for side in SIDES}

    data = _seated(model)
    saddle = model.geom(f"bike/{SADDLE_GEOM}").id
    trunk = model.body(SEAT_BODY).id
    for c in data.contact[: data.ncon]:
        bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
        if any(model.body(b).name.startswith("bike/") for b in bodies) and 0 not in bodies:
            assert {c.geom1, c.geom2} & {saddle} and trunk in bodies, (
                "in the seated pose only the pelvis touches the bike (on the saddle)"
            )


def test_k1_stays_seated_holding_its_pose(model):
    data = _seated(model)
    target = data.qpos.copy()
    _hold_pose(model, data, target, seconds=2.0)
    assert np.isfinite(data.qpos).all()
    assert seat_contact(model, data) > 20.0, "part of the K1's weight rests on the saddle"
    assert _clip_error(model, data) < 0.002
    frame = model.body("bike/frame").id
    assert data.xmat[frame].reshape(3, 3)[2, 2] > 0.99, "feet and wheels keep the bike upright"


def test_k1_can_rise_off_the_saddle_with_hands_still_clipped(model):
    """The seat is contact only: an upward push lifts the pelvis off and takes the bike along."""
    data = _seated(model)
    target = data.qpos.copy()
    _hold_pose(model, data, target, seconds=0.5)
    _hold_pose(model, data, target, seconds=0.5, push_up=250.0)
    assert seat_contact(model, data) == 0.0
    assert _pelvis_saddle_distance(model, data) > 0.01, "pelvis is clear of the saddle"
    assert _clip_error(model, data) < 0.005
    frame = model.body("bike/frame").id
    assert data.xpos[frame, 2] > 0.1, "the bike hangs from the clipped hands"


def test_seated_scene_rejects_other_bike_models():
    with pytest.raises(ValueError):
        k1_seated_bike_scene(bike_xml=PRIMITIVE_BIKE_XML)
