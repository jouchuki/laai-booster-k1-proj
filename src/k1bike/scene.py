"""Build MuJoCo scenes: the bike alone, or the K1 and the bike together."""

import math
from pathlib import Path

import mujoco
import numpy as np

from k1bike.paths import BIKE_XML, k1_xml


def _add_floor(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.5, 0.55, 0.6, 1],
    )


def bike_scene(bike_xml: Path = BIKE_XML) -> mujoco.MjSpec:
    """Bike on a floor, nothing else. For checking the bike model on its own."""
    spec = mujoco.MjSpec.from_file(str(bike_xml))
    _add_floor(spec)
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    return spec


def initialize_bike_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    prefix: str = "",
    forward_speed: float = 0.0,
    lean_degrees: float = 0.0,
    drop_height: float = 0.0,
) -> None:
    """Set the bike's initial state on freshly reset simulation data.

    Lean is about the bike's forward (local x) axis. Drop height is a vertical
    offset from the model's nominal pose; use a small positive offset when leaning
    to avoid starting with the tyre edges penetrating the floor. A forward push
    starts the wheels spinning at the corresponding rolling speed, rather than
    initially sliding locked wheels along the floor. Only the bike is changed in
    a combined K1/bike scene. No stabilizing controller or external support is added.
    """
    if not all(math.isfinite(value) for value in (forward_speed, lean_degrees, drop_height)):
        raise ValueError("Initial bike speed, lean and drop must be finite")
    if drop_height < 0:
        raise ValueError("Drop height must be nonnegative")

    root = model.joint(f"{prefix}root")
    qpos = int(model.jnt_qposadr[root.id])
    dof = int(model.jnt_dofadr[root.id])
    data.qpos[qpos : qpos + 7] = model.qpos0[qpos : qpos + 7]
    data.qpos[qpos + 2] += drop_height
    lean = np.zeros(4)
    mujoco.mju_axisAngle2Quat(lean, np.array([1.0, 0.0, 0.0]), math.radians(lean_degrees))
    nominal = model.qpos0[qpos + 3 : qpos + 7]
    mujoco.mju_mulQuat(data.qpos[qpos + 3 : qpos + 7], nominal, lean)
    direction = np.zeros(3)
    mujoco.mju_rotVecQuat(direction, np.array([1.0, 0.0, 0.0]), nominal)
    data.qvel[dof : dof + 6] = 0.0
    data.qvel[dof : dof + 3] = forward_speed * direction
    mujoco.mj_forward(model, data)
    for name in ("rear_wheel", "front_wheel"):
        wheel = model.joint(f"{prefix}{name}")
        geom = model.geom(f"{prefix}{name}")
        if model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = float(geom.size[0])
        else:
            # Mesh vertices are stored in the compiler's principal-axis frame.
            # Recover their distance to the world-space wheel axle, not an AABB
            # half-size (which need not be the radius for a compiled mesh).
            mesh = int(model.geom_dataid[geom.id])
            first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[first : first + count]
            world = vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
            offset = world - data.xanchor[wheel.id]
            axis = data.xaxis[wheel.id]
            radius = float(np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1).max())
        data.qvel[int(model.jnt_dofadr[wheel.id])] = forward_speed / radius
    mujoco.mj_forward(model, data)


def k1_bike_scene(bike_pos=(-0.6, 0.0, 0.0), bike_xml: Path = BIKE_XML) -> mujoco.MjSpec:
    """K1 (from booster_assets) plus the bike, on a floor.

    The K1 model is the base so its physics options and mesh paths are kept.
    Bike names get a "bike/" prefix. The robot is not yet attached to the bike;
    hand-to-grip and seat constraints are the next step (phase 3 of the plan).
    """
    spec = mujoco.MjSpec.from_file(str(k1_xml()))
    bike = mujoco.MjSpec.from_file(str(bike_xml))
    frame = spec.worldbody.add_frame(pos=list(bike_pos))
    frame.attach_body(bike.worldbody.first_body(), "bike/", "")
    _add_floor(spec)
    return spec
