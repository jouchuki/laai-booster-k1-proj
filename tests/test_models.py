"""Quick checks that the models load and simulate. Run before any long training job."""

import subprocess
import sys

import mujoco
import numpy as np
import pytest

from k1bike.paths import PRIMITIVE_BIKE_XML, PUKY_BIKE_XML, REPO_ROOT, YELLOW_BIKE_XML, k1_xml
from k1bike.scene import bike_scene, initialize_bike_state, k1_bike_scene


def _step(model: mujoco.MjModel, seconds: float, data: mujoco.MjData | None = None):
    data = data or mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


@pytest.fixture(
    params=[YELLOW_BIKE_XML, PRIMITIVE_BIKE_XML, PUKY_BIKE_XML],
    ids=["yellow_bike", "primitive_bike", "puky_bike"],
    scope="module",
)
def bike_path(request):
    return request.param


@pytest.fixture(scope="module")
def bike_model(bike_path):
    return bike_scene(bike_path).compile()


def test_k1_model_loads():
    model = mujoco.MjModel.from_xml_path(str(k1_xml()))
    assert model.nu == 22, "K1 should have 22 actuated joints"


def test_bike_rests_on_ground(bike_model):
    model = bike_model
    data = _step(model, 1.0)
    assert np.isfinite(data.qpos).all()
    # Origin sits on the ground under the rear axle: it must not sink or fly.
    assert abs(data.qpos[2]) < 0.01


def test_bike_rolls_when_pushed(bike_model):
    model = bike_model
    data = mujoco.MjData(model)
    initialize_bike_state(model, data, forward_speed=2.0)
    data = _step(model, 1.0, data)
    assert data.qpos[0] > 1.0, "bike should roll forward, not stick to the floor"


def test_bike_free_fall_matches_gravity(bike_model):
    model = bike_model
    data = mujoco.MjData(model)
    initialize_bike_state(model, data, drop_height=0.3)
    frame = model.body("frame").id
    height = float(data.subtree_com[frame, 2])
    data = _step(model, 0.1, data)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, "Free-fall check must happen before the wheels touch the floor"
    expected_drop = 0.5 * 9.81 * data.time**2
    # Allow one timestep of integration error relative to continuous free fall.
    assert height - data.subtree_com[frame, 2] == pytest.approx(
        expected_drop, abs=9.81 * data.time * model.opt.timestep
    )


@pytest.mark.parametrize("lean", [-5.0, 5.0])
def test_unsupported_bike_falls_from_a_lean(lean, bike_model):
    model = bike_model
    data = mujoco.MjData(model)
    initialize_bike_state(model, data, lean_degrees=lean, drop_height=0.03)
    frame = model.body("frame").id
    height = float(data.subtree_com[frame, 2])
    data = _step(model, 3.0, data)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.xmat[frame].reshape(3, 3)[2, 2] < 0.3, "The bike should end up on its side"
    assert data.subtree_com[frame, 2] < height - 0.1


def test_initial_push_is_rolling_instead_of_sliding(bike_model):
    model = bike_model
    data = mujoco.MjData(model)
    initialize_bike_state(model, data, forward_speed=2.0)
    for name in ("rear_wheel", "front_wheel"):
        geom = model.geom(name)
        body = int(model.geom_bodyid[geom.id])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        # All models start with their wheel axles above a z=0 floor. This
        # checks contact-point velocity, independently of the launcher's radius
        # calculation and of the collider's cylinder/mesh representation.
        contact_point = data.xanchor[model.joint(name).id].copy()
        contact_point[2] = 0.0
        contact_offset = contact_point - data.xipos[body]
        contact_velocity = velocity[3:] + np.cross(velocity[:3], contact_offset)
        assert contact_velocity[0] == pytest.approx(0.0, abs=1e-6)


def test_bike_has_grips_and_seat(bike_model):
    model = bike_model
    for site in ("left_grip", "right_grip", "seat"):
        model.site(site)  # raises KeyError if missing


def test_k1_and_bike_scene_simulates(bike_path):
    model = k1_bike_scene(bike_xml=bike_path).compile()
    assert model.nu == 22
    data = _step(model, 0.5)
    assert np.isfinite(data.qpos).all()


@pytest.mark.parametrize(
    "bike_xml, expected_wheelbase, expected_mass",
    [(YELLOW_BIKE_XML, 0.565, 1.5), (PUKY_BIKE_XML, 0.50, 3.7)],
)
def test_converted_wheelbase_and_configured_mass(bike_xml, expected_wheelbase, expected_mass):
    model = bike_scene(bike_xml).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    front = data.xanchor[model.joint("front_wheel").id]
    rear = data.xanchor[model.joint("rear_wheel").id]
    assert np.linalg.norm(front - rear) == pytest.approx(expected_wheelbase, abs=1e-7)
    assert front[2] == pytest.approx(rear[2], abs=1e-7)
    assert model.body_mass.sum() == pytest.approx(expected_mass)


@pytest.mark.parametrize(
    "bike_xml", [YELLOW_BIKE_XML, PUKY_BIKE_XML], ids=["yellow_bike", "puky_bike"]
)
def test_converted_steering_moves_front_assembly_without_moving_frame(bike_xml):
    model = bike_scene(bike_xml).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    front = model.body("front_wheel").id
    rear = model.body("rear_wheel").id
    frame = model.body("frame").id
    rear_before = data.xpos[rear].copy()
    frame_before = data.xmat[frame].copy()
    wheel_before = data.xmat[front].copy()
    data.qpos[model.jnt_qposadr[model.joint("steer").id]] = 0.4
    mujoco.mj_forward(model, data)
    steered_orientation = data.xmat[front].copy()
    assert not np.allclose(steered_orientation, wheel_before)
    data.qpos[model.jnt_qposadr[model.joint("front_wheel").id]] = 0.8
    mujoco.mj_forward(model, data)
    assert not np.allclose(data.xmat[front], steered_orientation)
    np.testing.assert_allclose(data.xpos[rear], rear_before)
    np.testing.assert_allclose(data.xmat[frame], frame_before)
    for geom in range(model.ngeom):
        if "_visual" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""):
            assert model.geom_contype[geom] == 0
            assert model.geom_conaffinity[geom] == 0


def test_yellow_bike_raised_handlebars_keep_saddle_and_grip_spacing():
    model = bike_scene(YELLOW_BIKE_XML).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left = data.site_xpos[model.site("left_grip").id]
    right = data.site_xpos[model.site("right_grip").id]
    assert left[2] == pytest.approx(0.45, abs=1e-7)
    assert right[2] == pytest.approx(0.45, abs=1e-7)
    assert np.linalg.norm(left - right) == pytest.approx(0.269670426, abs=1e-7)
    assert data.site_xpos[model.site("seat").id, 2] == pytest.approx(0.358265822, abs=1e-7)
    assert model.site_bodyid[model.site("left_grip").id] == model.body("fork").id
    assert model.site_bodyid[model.site("right_grip").id] == model.body("fork").id


def test_puky_bike_visuals_keep_texture_and_all_source_triangles():
    model = bike_scene(PUKY_BIKE_XML).compile()
    assert model.ntex == 1
    assert model.tex_width[0] > 0 and model.tex_height[0] > 0
    triangle_count = 0
    for body in ("frame", "fork", "rear_wheel", "front_wheel"):
        geom = model.geom(f"{body}_visual")
        assert model.geom_bodyid[geom.id] == model.body(body).id
        assert model.geom_matid[geom.id] >= 0
        mesh = int(model.geom_dataid[geom.id])
        assert model.mesh_texcoordnum[mesh] > 0
        triangle_count += int(model.mesh_facenum[mesh])
    assert triangle_count == 25000


def _mesh_world_vertices(model, data, geom_name):
    geom = model.geom(geom_name)
    mesh = int(model.geom_dataid[geom.id])
    first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[first : first + count]
    return vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]


def test_puky_bike_visual_and_collision_dimensions_match_owner_measurements():
    model = bike_scene(PUKY_BIKE_XML).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.site_xpos[model.site("seat").id, 2] == pytest.approx(0.42, abs=1e-7)
    # Check the actual mesh and collision surfaces as well as attachment sites.
    assert _mesh_world_vertices(model, data, "frame_visual")[:, 2].max() == pytest.approx(
        0.42, abs=1e-6
    )
    saddle = model.geom("saddle").id
    assert data.geom_xpos[saddle, 2] + model.geom_size[saddle, 2] == pytest.approx(0.42, abs=1e-7)
    for side in ("left", "right"):
        site = model.site(f"{side}_grip").id
        assert data.site_xpos[site, 2] == pytest.approx(0.53, abs=1e-7)
        assert model.site_bodyid[site] == model.body("fork").id
    for name in ("rear_wheel", "front_wheel"):
        axle = data.xanchor[model.joint(name).id]
        assert axle[2] == pytest.approx(0.11, abs=1e-7)
        for geom_name in (name, f"{name}_visual"):
            vertices = _mesh_world_vertices(model, data, geom_name)
            radius = np.linalg.norm((vertices - axle)[:, [0, 2]], axis=1).max()
            assert radius == pytest.approx(0.11, abs=1e-6)
        assert _mesh_world_vertices(model, data, name)[:, 2].min() == pytest.approx(0, abs=1e-6)


def test_puky_bike_converter_supports_scaled_asset_in_another_directory(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/convert_puky_bike.py"),
            "--output",
            str(tmp_path),
            "--wheelbase",
            "0.452",
            "--tyre-diameter",
            "0.20",
            "--saddle-height",
            "0.40",
            "--handlebar-height",
            "0.51",
            "--mass",
            "2.0",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    model = bike_scene(tmp_path / "bike.xml").compile()
    data = mujoco.MjData(model)
    initialize_bike_state(model, data, forward_speed=1.0)
    front = data.xanchor[model.joint("front_wheel").id]
    rear = data.xanchor[model.joint("rear_wheel").id]
    assert np.linalg.norm(front - rear) == pytest.approx(0.452, abs=1e-7)
    assert model.body_mass.sum() == pytest.approx(2.0)
    assert data.site_xpos[model.site("seat").id, 2] == pytest.approx(0.40, abs=1e-7)
    for side in ("left", "right"):
        assert data.site_xpos[model.site(f"{side}_grip").id, 2] == pytest.approx(0.51, abs=1e-7)
    assert front[2] == pytest.approx(0.10, abs=1e-7)
    assert model.ntex == 1
    assert _step(model, 0.5, data).qpos[0] > 0.3


@pytest.mark.parametrize(
    "bike_xml", [PUKY_BIKE_XML, YELLOW_BIKE_XML], ids=["puky_bike", "yellow_bike"]
)
def test_saddle_collision_follows_the_visible_saddle(bike_xml):
    """The flat saddle box is replaced by slices whose surface matches the visible saddle, so a
    rider rests on what the viewer shows (the Puky's saddle is 16 mm lower in the middle)."""
    from k1bike.saddle import saddle_geom_ids

    model = bike_scene(bike_xml).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    slices = set(saddle_geom_ids(model))
    box = model.geom("saddle").id
    assert box not in slices and len(slices) > 5, "the box no longer collides"
    assert model.body_mass.sum() == pytest.approx(bike_scene(bike_xml).compile().body_mass.sum())
    centre, half = data.geom_xpos[box], model.geom_size[box]
    collision, visible = (
        np.array([0, 0, 0, 1, 0, 0], np.uint8),
        np.array([0, 0, 1, 0, 0, 0], np.uint8),
    )
    hit, diffs = np.zeros(1, np.int32), []
    for x in np.linspace(centre[0] - 0.85 * half[0], centre[0] + 0.85 * half[0], 25):
        for y in np.linspace(centre[1] - 0.6 * half[1], centre[1] + 0.6 * half[1], 5):
            start = np.array([x, y, centre[2] + 0.2])
            a = mujoco.mj_ray(model, data, start, np.array([0, 0, -1.0]), collision, 1, -1, hit)
            on_slice = hit[0] in slices
            b = mujoco.mj_ray(model, data, start, np.array([0, 0, -1.0]), visible, 1, -1, hit)
            if a >= 0 and b >= 0 and on_slice and abs(b - a) < 0.05:
                diffs.append(b - a)
    assert len(diffs) > 50
    assert np.percentile(np.abs(diffs), 90) < 0.002
    assert np.max(np.abs(diffs)) < 0.004


@pytest.mark.parametrize(
    "bike_xml", [PUKY_BIKE_XML, YELLOW_BIKE_XML], ids=["puky_bike", "yellow_bike"]
)
def test_nothing_pokes_up_through_the_saddle(bike_xml):
    """No other collision geom (such as the seat post) reaches above the saddle's dip inside the
    saddle's footprint, so a rider rests on the saddle surface, not on the post."""
    from k1bike.saddle import saddle_geom_ids

    model = bike_scene(bike_xml).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = model.geom("saddle").id
    centre, half = data.geom_xpos[box], model.geom_size[box]
    dip = data.site_xpos[model.site("saddle_dip").id]
    saddle = set(saddle_geom_ids(model))
    collision = np.array([0, 0, 0, 1, 0, 0], np.uint8)
    hit = np.zeros(1, np.int32)
    for x in np.linspace(centre[0] - half[0], centre[0] + half[0], 41):
        for y in np.linspace(centre[1] - half[1], centre[1] + half[1], 11):
            start = np.array([x, y, dip[2] + 0.1])
            dist = mujoco.mj_ray(model, data, start, np.array([0, 0, -1.0]), collision, 1, -1, hit)
            if dist >= 0 and hit[0] not in saddle and model.geom_bodyid[hit[0]] != 0:
                assert start[2] - dist < dip[2] - 0.003, model.geom(hit[0]).name


def test_k1_bike_scene_has_one_floor():
    model = k1_bike_scene().compile()
    planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert [model.geom(g).name for g in planes] == ["floor"]
