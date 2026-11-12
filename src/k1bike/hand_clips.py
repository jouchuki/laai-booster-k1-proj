"""Orange clip geometry and fixed hand attachments for MuJoCo."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

ORANGE = [1.0, 0.27, 0.025, 1.0]
SIDES = (("left", 1), ("right", -1))


@dataclass(frozen=True)
class ClipDimensions:
    total_length_m: float = 0.10
    handle_centre_s_m: float = 0.02
    hand_section_start_s_m: float = 0.04
    hand_centre_s_m: float = 0.06
    wrist_start_s_m: float = 0.08
    hand_diameter_m: float = 0.04
    handle_diameter_upper_estimate_m: float = 0.03

    @property
    def centre_spacing_m(self):
        return self.hand_centre_s_m - self.handle_centre_s_m

    @property
    def wrist_to_hand_m(self):
        return self.wrist_start_s_m - self.hand_centre_s_m


DEFAULT_DIMENSIONS = ClipDimensions()
HAND_LOCAL_M = 0.208  # Estimated hand frame in the terminal arm mesh.


def matrix_quaternion(matrix):
    value = np.zeros(4)
    mujoco.mju_mat2Quat(value, np.asarray(matrix).ravel())
    return value


def add_orange_clips(spec, dimensions=DEFAULT_DIMENSIONS):
    """Add clips with 4 cm rearward spacing, equal height and inactive welds."""
    fork = spec.body("bike/fork")
    for side, sign in SIDES:
        arm = spec.body(f"{side}_elbow_yaw_link")
        arm.add_site(
            name=f"{side}_hand_centre",
            pos=[0, sign * HAND_LOCAL_M, 0],
            size=[0.004] * 3,
            group=4,
        )
        arm.add_site(
            name=f"{side}_wrist_reference",
            pos=[0, sign * (HAND_LOCAL_M - dimensions.wrist_to_hand_m), 0],
            size=[0.003] * 3,
            group=4,
        )
        grip = np.asarray(spec.site(f"bike/{side}_grip").pos).copy()
        hand = grip + [-dimensions.centre_spacing_m, 0, 0]
        fork.add_site(name=f"{side}_clip_hand_target", pos=hand, size=[0.004] * 3, group=4)
        start = grip + [dimensions.handle_centre_s_m, 0, 0]
        end = start + [-dimensions.total_length_m, 0, 0]
        fork.add_geom(
            name=f"{side}_clip_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(start + end) / 2 + [0, 0, -0.015],
            size=[dimensions.total_length_m / 2, 0.012, 0.003],
            rgba=ORANGE,
            contype=0,
            conaffinity=0,
            mass=0,
            group=2,
        )
        # Diameters locate ring centre-lines; opening size is unmeasured.
        for kind, centre, radius, half_length in (
            ("handle", grip, dimensions.handle_diameter_upper_estimate_m / 2, 0.009),
            ("hand", hand, dimensions.hand_diameter_m / 2, 0.02),
        ):
            for index in range(20):
                angles = np.array([index, index + 1]) * 2 * np.pi / 20
                if kind == "handle":
                    ring = np.column_stack(
                        (radius * np.cos(angles), np.zeros(2), radius * np.sin(angles))
                    )
                else:
                    ring = np.column_stack(
                        (np.zeros(2), radius * np.cos(angles), radius * np.sin(angles))
                    )
                for offset in (-half_length, half_length):
                    shift = [0, offset, 0] if kind == "handle" else [offset, 0, 0]
                    fork.add_geom(
                        name=f"{side}_{kind}_collar_{index}_{offset:g}",
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        fromto=(ring + centre + shift).ravel(),
                        size=[0.002],
                        rgba=ORANGE,
                        contype=0,
                        conaffinity=0,
                        mass=0,
                        group=2,
                    )
        spec.add_equality(
            name=f"{side}_fixed_clip",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{side}_hand_centre",
            name2=f"{side}_clip_hand_target",
            active=False,
            solref=[0.004, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )
    return asdict(dimensions)


def lock_fitted_clips(spec, model, data):
    """Align weld orientations and activate them after fitting the hand positions."""
    mujoco.mj_forward(model, data)
    fork = model.body("bike/fork").id
    fork_rotation = data.xmat[fork].reshape(3, 3)
    for side, _ in SIDES:
        hand = model.site(f"{side}_hand_centre").id
        target_name = f"{side}_clip_hand_target"
        target = model.site(target_name).id
        error = np.linalg.norm(data.site_xpos[hand] - data.site_xpos[target])
        if error > 0.001:
            raise ValueError(f"{side} hand is {error:.4f} m from the clip; fit the pose first")
        orientation = fork_rotation.T @ data.site_xmat[hand].reshape(3, 3)
        quaternion = matrix_quaternion(orientation)
        spec.site(target_name).quat = quaternion
        model.site_quat[target] = quaternion
        name = f"{side}_fixed_clip"
        spec.equality(name).active = True
        equality = model.equality(name).id
        model.eq_active0[equality] = True
        data.eq_active[equality] = True
    mujoco.mj_forward(model, data)
