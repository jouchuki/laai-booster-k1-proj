"""K1 seated on a bike: hands welded to the grips, pelvis resting on the saddle by contact.

Every test runs on each bike that has a solved seated pose (models/k1_on_bike/<bike>.json)."""

import mujoco
import numpy as np
import pytest

from k1bike.paths import BIKES, DEFAULT_BIKE, seated_pose_json
from k1bike.robots.k1 import hold_pose_torques
from k1bike.saddle import saddle_geom_ids
from k1bike.scene import bike_scene, k1_seated_bike_scene
from k1bike.seat import FORK_BODY, HAND_LINKS, SEAT_BODY, SIDES, seat_contact

SEATED_BIKES = [name for name, xml in BIKES.items() if seated_pose_json(xml).is_file()]


@pytest.fixture(scope="module", params=SEATED_BIKES)
def model(request):
    return k1_seated_bike_scene(bike_xml=BIKES[request.param]).compile()


def _bike_collision_geoms(bike_xml):
    """Colliding geoms of the bike as every scene loads it (saddle fitted to the visible one)."""
    bike = bike_scene(bike_xml).compile()
    frame = bike.body("frame").id
    return sum(
        bool(bike.geom_contype[g] or bike.geom_conaffinity[g])
        for g in range(bike.ngeom)
        if bike.geom_bodyid[g] != 0 and bike.body_rootid[bike.geom_bodyid[g]] == frame
    )


def test_the_puky_bike_has_a_seated_pose_and_is_the_default():
    assert DEFAULT_BIKE == "puky_bike"
    assert DEFAULT_BIKE in SEATED_BIKES
    default = k1_seated_bike_scene().compile()
    real = k1_seated_bike_scene(bike_xml=BIKES[DEFAULT_BIKE]).compile()
    assert default.nbody == real.nbody and np.allclose(
        default.key("seated").qpos, real.key("seated").qpos
    )


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
    return min(
        mujoco.mj_geomDistance(model, data, pelvis, g, 1.0, np.zeros(6))
        for g in saddle_geom_ids(model, "bike/")
    )


def _visible_gap(model, data):
    """Smallest vertical gap between the K1's visible meshes (group 1) and the bike's visible
    saddle (group 2) over the saddle, by rays against the real meshes: what the viewer shows."""
    box = model.geom("bike/saddle").id
    centre, half = data.geom_xpos[box], model.geom_size[box]
    robot, bike = np.array([0, 1, 0, 0, 0, 0], np.uint8), np.array([0, 0, 1, 0, 0, 0], np.uint8)
    hit, gaps = np.zeros(1, np.int32), []
    for x in np.linspace(centre[0] - half[0], centre[0] + half[0], 31):
        for y in np.linspace(centre[1] - half[1], centre[1] + half[1], 11):
            start = np.array([x, y, centre[2] - 0.15])
            up = mujoco.mj_ray(model, data, start, np.array([0, 0, 1.0]), robot, 1, -1, hit)
            if up < 0:
                continue
            bottom = start + [0, 0, up]
            down = mujoco.mj_ray(model, data, bottom, np.array([0, 0, -1.0]), bike, 1, -1, hit)
            if down >= 0:
                gaps.append(down)
    return min(gaps)


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


@pytest.mark.parametrize("bike", SEATED_BIKES)
def test_robot_collides_with_itself_and_the_bike(bike):
    """No passing through: the K1 keeps all its self-collisions and touches every bike
    collision geom, except each welded forearm and the fork (its hand overlaps the bar)."""
    model = k1_seated_bike_scene(bike_xml=BIKES[bike]).compile()
    blocked, self_pairs, bike_geoms = _blocked_bike_bodies(model)
    assert self_pairs == 174, "same self-collision as the walk task"
    assert bike_geoms == _bike_collision_geoms(BIKES[bike]), "every bike collision geom kept"
    assert blocked == {(HAND_LINKS[side], f"bike/{FORK_BODY}") for side in SIDES}

    data = _seated(model)
    saddle = set(saddle_geom_ids(model, "bike/"))
    trunk = model.body(SEAT_BODY).id
    for c in data.contact[: data.ncon]:
        bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
        if any(model.body(b).name.startswith("bike/") for b in bodies) and 0 not in bodies:
            assert {c.geom1, c.geom2} & saddle and trunk in bodies, (
                "in the seated pose only the pelvis touches the bike (on the saddle)"
            )


def test_k1_stays_seated_holding_its_pose(model):
    data = _seated(model)
    target = data.qpos.copy()
    _hold_pose(model, data, target, seconds=2.0)
    assert np.isfinite(data.qpos).all()
    assert seat_contact(model, data) > 60.0, "a large part of the K1's 193 N rests on the saddle"
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


def test_seated_scene_needs_a_solved_pose_for_its_bike():
    unsolved = [name for name in BIKES if name not in SEATED_BIKES]
    assert unsolved, "keep one bike without a pose to test this"
    with pytest.raises(FileNotFoundError, match="solve_seated_pose.py --bike"):
        k1_seated_bike_scene(bike_xml=BIKES[unsolved[0]])


def test_k1_visibly_sits_on_the_saddle(model):
    """No hovering: the visible pelvis rests on the visible saddle, at the start and while
    holding the pose (the saddle's collision follows the visible saddle, k1bike/saddle.py)."""
    data = _seated(model)
    assert _visible_gap(model, data) < 0.003
    _hold_pose(model, data, data.qpos.copy(), seconds=2.0)
    assert _visible_gap(model, data) < 0.003


def _load_split(model, seconds=4.0):
    """Average vertical load on the saddle, the feet and the hands (by balance) while holding
    the seated pose, after 1 s to settle, as fractions of the K1's weight."""
    data = _seated(model)
    target = data.qpos.copy()
    _hold_pose(model, data, target, seconds=1.0)
    saddle_ids = set(saddle_geom_ids(model, "bike/"))
    floor = model.geom("floor").id
    weight = model.body_subtreemass[model.body("trunk").id] * 9.81
    force, totals = np.zeros(6), {"saddle": [], "feet": [], "other": []}
    for step in range(int(seconds / model.opt.timestep)):
        hold_pose_torques(model, data, target)
        mujoco.mj_step(model, data)
        if step % 10:
            continue
        load = dict.fromkeys(totals, 0.0)
        for i, c in enumerate(data.contact[: data.ncon]):
            bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if all(b == 0 or model.body(b).name.startswith("bike/") for b in bodies):
                continue  # bike-floor or bike-bike contact
            mujoco.mj_contactForce(model, data, i, force)
            vertical = force[0] * abs(c.frame[2])
            if floor in (c.geom1, c.geom2):
                load["feet"] += vertical
            elif saddle_ids & {c.geom1, c.geom2}:
                load["saddle"] += vertical
            else:
                load["other"] += vertical
        for key, value in load.items():
            totals[key].append(value)
    split = {key: np.mean(values) / weight for key, values in totals.items()}
    split["hands"] = 1.0 - sum(split.values())
    return split


def test_weight_rests_mostly_on_the_saddle(model):
    """Like a rider on a balance bike: most weight on the saddle, the rest on the feet, and the
    hands only resting on the handlebar."""
    split = _load_split(model)
    assert split["saddle"] > 0.6, split
    assert split["feet"] > 0.05, f"feet rest on the floor: {split}"
    assert split["hands"] < 0.15, f"hands only rest on the bar: {split}"
    assert split["other"] < 0.01, f"nothing else touches the bike: {split}"


def test_k1_sits_in_the_saddle_dip(model):
    """The pelvis sits in the lowest part of the saddle's middle, and stays there."""
    data = _seated(model)
    _hold_pose(model, data, data.qpos.copy(), seconds=3.0)
    saddle_ids = set(saddle_geom_ids(model, "bike/"))
    trunk = model.body(SEAT_BODY).id
    xs = [
        c.pos[0]
        for c in data.contact[: data.ncon]
        if saddle_ids & {c.geom1, c.geom2}
        and trunk in {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
    ]
    assert xs, "the pelvis touches the saddle"
    dip = data.site_xpos[model.site("bike/saddle_dip").id]
    assert abs(np.mean(xs) - dip[0]) < 0.015
