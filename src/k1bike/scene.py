"""Build MuJoCo scenes: the bike alone, or the K1 and the bike together."""

import mujoco

from k1bike.paths import BIKE_XML, k1_xml


def _add_floor(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.5, 0.55, 0.6, 1],
    )


def bike_scene() -> mujoco.MjSpec:
    """Bike on a floor, nothing else. For checking the bike model on its own."""
    spec = mujoco.MjSpec.from_file(str(BIKE_XML))
    _add_floor(spec)
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    return spec


def k1_bike_scene(bike_pos=(-0.6, 0.0, 0.0)) -> mujoco.MjSpec:
    """K1 (from booster_assets) plus the bike, on a floor.

    The K1 model is the base so its physics options and mesh paths are kept.
    Bike names get a "bike/" prefix. The robot is not yet attached to the bike;
    hand-to-grip and seat constraints are the next step (phase 3 of the plan).
    """
    spec = mujoco.MjSpec.from_file(str(k1_xml()))
    bike = mujoco.MjSpec.from_file(str(BIKE_XML))
    frame = spec.worldbody.add_frame(pos=list(bike_pos))
    frame.attach_body(bike.worldbody.first_body(), "bike/", "")
    _add_floor(spec)
    return spec
