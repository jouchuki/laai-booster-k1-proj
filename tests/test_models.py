"""Quick checks that the models load and simulate. Run before any long training job."""

import mujoco
import numpy as np

from k1bike.paths import k1_xml
from k1bike.scene import bike_scene, k1_bike_scene


def _step(model: mujoco.MjModel, seconds: float, data: mujoco.MjData | None = None):
    data = data or mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def test_k1_model_loads():
    model = mujoco.MjModel.from_xml_path(str(k1_xml()))
    assert model.nu == 22, "K1 should have 22 actuated joints"


def test_bike_rests_on_ground():
    model = bike_scene().compile()
    data = _step(model, 1.0)
    assert np.isfinite(data.qpos).all()
    # Origin sits on the ground under the rear axle: it must not sink or fly.
    assert abs(data.qpos[2]) < 0.01


def test_bike_rolls_when_pushed():
    model = bike_scene().compile()
    data = mujoco.MjData(model)
    data.qvel[0] = 2.0
    data = _step(model, 1.0, data)
    assert data.qpos[0] > 1.0, "bike should roll forward, not stick to the floor"


def test_bike_has_grips_and_seat():
    model = bike_scene().compile()
    for site in ("left_grip", "right_grip", "seat"):
        model.site(site)  # raises KeyError if missing


def test_k1_and_bike_scene_simulates():
    model = k1_bike_scene().compile()
    assert model.nu == 22
    data = _step(model, 0.5)
    assert np.isfinite(data.qpos).all()
