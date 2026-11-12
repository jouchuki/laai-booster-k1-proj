"""Quick checks that the models load and simulate. Run before any long training job."""

import mujoco
import numpy as np
import pytest

from k1bike.paths import BIKE_XML, PRIMITIVE_BIKE_XML, k1_xml
from k1bike.scene import bike_scene, initialize_bike_state, k1_bike_scene


def _step(model: mujoco.MjModel, seconds: float, data: mujoco.MjData | None = None):
    data = data or mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


@pytest.fixture(
    params=[BIKE_XML, PRIMITIVE_BIKE_XML], ids=["bike", "bike_primitive"], scope="module"
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
        # Both models start with their wheel axles above a z=0 floor. This
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


def test_bike_wheelbase_matches_reported_measurement():
    model = bike_scene(BIKE_XML).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    front = data.xanchor[model.joint("front_wheel").id]
    rear = data.xanchor[model.joint("rear_wheel").id]
    assert np.linalg.norm(front - rear) == pytest.approx(0.565, abs=1e-7)
    assert front[2] == pytest.approx(rear[2], abs=1e-7)
    assert model.body_mass.sum() == pytest.approx(1.5), (
        "Provisional mass, not inferred from visuals"
    )


def test_bike_steering_moves_front_assembly_without_moving_frame():
    model = bike_scene(BIKE_XML).compile()
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
        if "_visual_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""):
            assert model.geom_contype[geom] == 0
            assert model.geom_conaffinity[geom] == 0


def test_bike_raised_handlebars_keep_saddle_and_grip_spacing():
    model = bike_scene(BIKE_XML).compile()
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
