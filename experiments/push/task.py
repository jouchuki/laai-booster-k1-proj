"""Walk the bike along a straight line at a commanded speed, on the unsupported bike.

The K1 sits on the Puky as in the free-bike task (experiments/sit_free/task.py): seated pose,
hands welded to the grips, walk-policy interface (690 inputs, 20 joint actions). The bike is
free: nothing holds it upright or on course, and its steering turns under the robot's hands.
The velocity command (always zero while sitting) now carries a forward speed. The robot is
rewarded for pushing the bike along at that speed, on the line it started on. The saddle's
height and fore-aft position differ a little in every episode.

Three tasks share everything except their rewards, to compare a minimal reward with shaped ones:

  * LAAI-K1-Push-Puky-Simple: only the task (speed, straight line, stay seated, do not fall,
    smooth actions).
  * LAAI-K1-Push-Puky-Shaped: the same plus terms for the gait, posture and motor load.
  * LAAI-K1-Push-Puky-Form: the shaped rewards plus four terms for the riding form: trunk
    facing along the bike, legs straight or slightly outward, no body part hitting the bike
    (except the pelvis on the saddle), and left and right steps alike, half a step apart.

Their returns are not comparable (different reward sums); compare the logged metrics and the
final evaluation (experiments/push/evaluate.py), which are identical for all three.

A fourth task steers, like the walk policy following a joystick:

  * LAAI-K1-Push-Puky-Steer: the Form rewards, but the command also carries a turn rate, and
    the straight-line term is replaced by tracking that turn rate on the bike's yaw rate.
    The turn rate is capped at MAX_CURVATURE times the commanded speed: the bike only turns
    while it rolls, and the arms (hands welded to the grips) turn the bar about 17 deg at
    most before nearing the torque limit (see experiments/push/README.md).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, fields

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, wrap_to_pi

from experiments.push.runner import PushRunner, PushSteerRunner
from experiments.sit import task as sit
from experiments.sit_free import task as free
from experiments.sit_free.runner import PolicyInitRunnerCfg
from experiments.sit_free.task import saddle_vertical_load_fraction
from k1bike.robots.k1 import DEPLOY_JOINTS, EFFORT, FOOT_SITES, POLICY_JOINT_REGEX

SIMPLE_TASK_ID = "LAAI-K1-Push-Puky-Simple"
SHAPED_TASK_ID = "LAAI-K1-Push-Puky-Shaped"
FORM_TASK_ID = "LAAI-K1-Push-Puky-Form"
STEER_TASK_ID = "LAAI-K1-Push-Puky-Steer"
COMMAND = "twist"  # the walk task's command name, so mjlab's velocity terms work unchanged

# --- Saddle randomization (per episode) --------------------------------------------------------
# The real saddle and the robot's place on it are never exactly the model's: the saddle's
# collision surface moves by up to this much, along the bike (x) and up or down (z).
SADDLE_SHIFT_X = 0.02  # m, forward or back
SADDLE_SHIFT_Z = 0.015  # m, higher or lower

# --- The speed command -------------------------------------------------------------------------
STANDING_SHARE = 0.2  # share of commands that are zero: sit still
# Non-zero commands start at 0.15 m/s: below that, standing still already earns much of the
# tracking reward (SPEED_STD). Widened once at iteration 1500, as the walk task does.
SPEED_STAGES = [
    {"step": 0, "lin_vel_x": (0.15, 0.3)},
    {"step": 1500 * 24, "lin_vel_x": (0.15, 0.6)},
]
SPEED_STD = 0.15  # m/s: standing still on a 0.3 m/s command earns 2% of the tracking reward
SPEED_FILTER_S = 0.5  # each push-off surges and fades; track the speed averaged over this time
MOVING = 0.05  # m/s: commands above this count as "moving" in the shaped rewards
TORQUE_SOFT_LIMIT = 0.8  # share of a joint's torque limit that costs nothing

# --- The turn-rate command (the Steer task) ----------------------------------------------------
# A rolling bike turns at about speed * tan(steer * cos 18 deg) / 0.5 m (wheelbase 0.5 m, steering
# axis 18 deg from vertical). The arms turn the bar about 17 deg before a shoulder nears its torque
# limit, so the turn rate is capped at this curvature (1/m) times the commanded speed: a 2 m
# radius, about 15 deg of steering. Sitting still means no turning either.
MAX_CURVATURE = 0.5
# Turn rates start at +-0.15 rad/s (the cap at 0.3 m/s, the first stage's top speed) and widen
# with the speed at iteration 1500, as the walk task's command does.
STEER_STAGES = [
    {**SPEED_STAGES[0], "ang_vel_z": (-0.15, 0.15)},
    {**SPEED_STAGES[1], "ang_vel_z": (-0.3, 0.3)},
]
TURN_STD = 0.1  # rad/s: not turning on a 0.15 rad/s command earns 11% of the tracking reward
TURN_FILTER_S = 0.5  # as SPEED_FILTER_S: each push-off also rocks the bike's heading

# --- Riding form (the Form task) ---------------------------------------------------------------
# Legs: how far each knee and foot is outward of its hip, across the bike (m; positive is away
# from the bike's centre line on that leg's side). Seated: knee +1.3 cm, foot +3.4 cm. Inside
# these bands costs nothing: inward risks hitting the bike, wider than this is splayed.
KNEE_OUT_BAND = (0.0, 0.04)
FOOT_OUT_BAND = (0.0, 0.08)
LEG_BAND_SCALE = 0.02  # m outside a band that costs 1
LEG_BODIES = {"hip": "hip_roll_link", "knee": "knee_pitch_link", "foot": "ankle_roll_link"}
# Step symmetry: a step's stats only count for this long after that foot's last touchdown.
STEP_STALE_S = 2.0
STEP_PITCH_JOINTS = ("hip_pitch", "knee_pitch", "ankle_pitch")


def _line(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The bike's position along and across its starting line (m), and its heading (rad)."""
    bike = env.scene["bike"].data
    pos = bike.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return pos[:, 0], pos[:, 1], bike.heading_w


def bike_distance(env) -> torch.Tensor:
    """How far the bike has travelled along its starting line this episode (m)."""
    return _line(env)[0]


def lateral_deviation(env) -> torch.Tensor:
    """Sideways distance from the starting line (m)."""
    return _line(env)[1].abs()


def heading_error_deg(env) -> torch.Tensor:
    return torch.rad2deg(_line(env)[2].abs())


def bike_speed(env) -> torch.Tensor:
    """The bike's forward speed, along its own heading (m/s)."""
    return env.scene["bike"].data.root_link_lin_vel_b[:, 0]


def bike_line_obs(env) -> torch.Tensor:
    """Critic only: sideways offset from the line and heading (the robot cannot sense them)."""
    _, across, heading = _line(env)
    return torch.stack([across, heading], dim=-1)


def speed_command(env) -> torch.Tensor:
    return env.command_manager.get_command(COMMAND)[:, 0]


def turn_command(env) -> torch.Tensor:
    """The commanded turn rate (rad/s, positive to the left); always zero in the straight tasks."""
    return env.command_manager.get_command(COMMAND)[:, 2]


def bike_yaw_rate(env) -> torch.Tensor:
    """The bike's turn rate about its own vertical axis (rad/s, positive to the left)."""
    return env.scene["bike"].data.root_link_ang_vel_b[:, 2]


# --- The turn-rate command and heading (the Steer task) ----------------------------------------
class BikeTurnCommand(UniformVelocityCommand):
    """mjlab's velocity command, with the turn rate capped at max_curvature x |speed|."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        cap = self.cfg.max_curvature * self.vel_command_b[env_ids, 0].abs()
        self.vel_command_b[env_ids, 2] = torch.maximum(
            torch.minimum(self.vel_command_b[env_ids, 2], cap), -cap
        )


@dataclass(kw_only=True)
class BikeTurnCommandCfg(UniformVelocityCommandCfg):
    max_curvature: float = MAX_CURVATURE

    def build(self, env) -> BikeTurnCommand:
        return BikeTurnCommand(self, env)


class CommandedHeading:
    """The heading the commands lead to (their turn rate integrated since the episode began),
    against the bike's. Updated once per control step, however many terms ask."""

    def __init__(self, env):
        self.step = -1
        self.target = torch.zeros(env.num_envs, device=env.device)

    @staticmethod
    def of(env, update: bool = True) -> CommandedHeading:
        tracker = getattr(env, "_laai_heading", None)
        if tracker is None:
            tracker = env._laai_heading = CommandedHeading(env)
        if update and env.common_step_counter != tracker.step:
            tracker.step = env.common_step_counter
            tracker.target += env.step_dt * turn_command(env)
        return tracker

    def reset(self, env_ids=None) -> None:
        self.target[slice(None) if env_ids is None else env_ids] = 0.0

    def error(self, env) -> torch.Tensor:
        """Bike heading minus the commanded one, in [-pi, pi] (rad). Every episode starts
        heading along the world's x axis, as the straight tasks' line does."""
        return wrap_to_pi(env.scene["bike"].data.heading_w - self.target)


# --- Rewards -----------------------------------------------------------------------------------
class track_bike_speed:
    """exp(-((command - bike speed) / std)^2), on the speed averaged over about `tau` s."""

    def __init__(self, cfg: RewardTermCfg, env):
        self.filtered = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.filtered[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, std: float, tau: float) -> torch.Tensor:
        self.filtered += min(1.0, env.step_dt / tau) * (bike_speed(env) - self.filtered)
        return torch.exp(-torch.square((speed_command(env) - self.filtered) / std))


class track_bike_turn_rate:
    """exp(-((command - bike yaw rate) / std)^2), on the yaw rate averaged over about `tau` s.

    Also resets the commanded heading (CommandedHeading) that the critic and metrics read.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        self.env = env
        self.filtered = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        self.filtered[slice(None) if env_ids is None else env_ids] = 0.0
        CommandedHeading.of(self.env, update=False).reset(env_ids)

    def __call__(self, env, std: float, tau: float) -> torch.Tensor:
        CommandedHeading.of(env)
        self.filtered += min(1.0, env.step_dt / tau) * (bike_yaw_rate(env) - self.filtered)
        return torch.exp(-torch.square((turn_command(env) - self.filtered) / std))


def bike_heading_obs(env) -> torch.Tensor:
    """Critic only (Steer task): bike heading minus the commanded heading (rad)."""
    return CommandedHeading.of(env).error(env).unsqueeze(-1)


def on_line(env, std_m: float, std_rad: float) -> torch.Tensor:
    """1 on the starting line and heading along it; less when drifting or turning away."""
    _, across, heading = _line(env)
    return torch.exp(-torch.square(across / std_m) - torch.square(heading / std_rad))


def saddle_load(env, target: float) -> torch.Tensor:
    """1 when at least `target` of the K1's weight rests vertically on the saddle."""
    return (saddle_vertical_load_fraction(env) / target).clamp(max=1.0)


def _feet_down(env) -> torch.Tensor:
    return env.scene.sensors["feet_floor_contact"].data.found > 0


def both_feet_off(env) -> torch.Tensor:
    return (~_feet_down(env).any(-1)).float()


def feet_on_floor_when_still(env) -> torch.Tensor:
    """Both feet down, counted only while the command is to sit still."""
    return sit.feet_on_floor(env) * (speed_command(env).abs() < MOVING).float()


def torque_ratio(env) -> torch.Tensor:
    """Each motor's torque as a share of its limit [worlds, motors]."""
    robot = env.scene["robot"]
    limit = getattr(env, "_laai_effort_limit", None)
    if limit is None:
        by_joint = dict(zip(DEPLOY_JOINTS, EFFORT, strict=True))
        limit = torch.tensor(
            [by_joint[name.rsplit("/", 1)[-1]] for name in robot.actuator_names],
            device=env.device,
        )
        env._laai_effort_limit = limit
    return robot.data.actuator_force.abs() / limit


def torque_near_limit(env, soft: float) -> torch.Tensor:
    """Squared torque above `soft` of each limit; one motor at its limit costs 1."""
    excess = (torque_ratio(env) - soft).clamp(min=0.0) / (1.0 - soft)
    return excess.square().sum(-1)


# --- Riding form: trunk along the bike, legs, collisions, step symmetry -------------------------
def _seated_trunk_forward() -> torch.Tensor:
    """The trunk's forward (x) axis in the bike's frame when seated: about 13 deg nose-down."""
    quat = torch.tensor(sit.POSE.trunk_quat, dtype=torch.float32).unsqueeze(0)
    return matrix_from_quat(quat)[0, :, 0]


SEATED_TRUNK_FORWARD = _seated_trunk_forward()


def trunk_yaw_pitch_to_bike(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Trunk yaw and pitch relative to the bike, minus their seated values (rad).

    Roll is left out: leaning the trunk against the bike's lean is how a rider balances, and
    the world-frame trunk_orientation term already covers it.
    """
    bike = env.scene["bike"].data
    trunk_forward = matrix_from_quat(env.scene["robot"].data.root_link_quat_w)[..., 0]
    f = quat_apply_inverse(bike.root_link_quat_w, trunk_forward)  # in the bike's frame
    f0 = SEATED_TRUNK_FORWARD.to(f.device)
    yaw = torch.atan2(f[:, 1], f[:, 0]) - torch.atan2(f0[1], f0[0])
    pitch = torch.atan2(f[:, 2], f[:, :2].norm(dim=-1)) - torch.atan2(f0[2], f0[:2].norm())
    return yaw, pitch


def trunk_to_bike(env, std_yaw: float, std_pitch: float) -> torch.Tensor:
    """1 with the trunk facing along the bike and pitched as when seated."""
    yaw, pitch = trunk_yaw_pitch_to_bike(env)
    return torch.exp(-torch.square(yaw / std_yaw) - torch.square(pitch / std_pitch))


def legs_outward(env) -> tuple[torch.Tensor, torch.Tensor]:
    """How far each knee and foot is outward of its hip across the bike (m), [worlds, 2 legs].

    Measured in the bike's frame, so a leaning or turning bike does not change it.
    """
    robot, bike = env.scene["robot"], env.scene["bike"].data
    ids = getattr(env, "_laai_leg_body_ids", None)
    if ids is None:
        ids = {
            part: [robot.find_bodies(f"^{side}_{body}$")[0][0] for side in sit.SIDES]
            for part, body in LEG_BODIES.items()
        }
        env._laai_leg_body_ids = ids
    origin, quat = bike.root_link_pos_w, bike.root_link_quat_w

    def across(part: str) -> torch.Tensor:
        pos = robot.data.body_link_pos_w[:, ids[part]] - origin.unsqueeze(1)
        return quat_apply_inverse(quat.unsqueeze(1).expand(-1, 2, -1), pos)[..., 1]

    side = torch.tensor([1.0, -1.0], device=env.device)  # left is +y, right is -y
    hip = across("hip")
    return (across("knee") - hip) * side, (across("foot") - hip) * side


def _outside(value: torch.Tensor, band: tuple[float, float]) -> torch.Tensor:
    return (band[0] - value).clamp(min=0.0) + (value - band[1]).clamp(min=0.0)


def leg_band(env, knee_band, foot_band, scale: float) -> torch.Tensor:
    """Squared distance outside the knee and foot bands, per `scale` m, summed over both legs."""
    knee, foot = legs_outward(env)
    excess = torch.cat([_outside(knee, knee_band), _outside(foot, foot_band)], dim=-1)
    return torch.square(excess / scale).sum(-1)


def bike_collision_force(env) -> torch.Tensor:
    """Force of the robot on the bike outside the saddle, as a share of its weight.

    The forearms never touch the fork in the simulation (their contacts are excluded: the
    hand welds define how hands and bar interact), so holding the bar is free.
    """
    force = env.scene.sensors["bike_collision"].data.force  # [worlds, bike geoms, 3]
    return force.norm(dim=-1).sum(-1) / sit.ROBOT_WEIGHT_N


def bike_collision_any(env) -> torch.Tensor:
    return (env.scene.sensors["bike_collision"].data.found > 0).any(-1).float()


def bike_collision(env) -> torch.Tensor:
    """1 per step with any contact, plus the force as a share of the weight (at most 1 more)."""
    return bike_collision_any(env) + bike_collision_force(env).clamp(max=1.0)


class StepTracker:
    """Each foot's latest completed step, for comparing left and right.

    Per foot: air time of the last swing, mean floor force of the last stance (share of the
    weight), range of each pitch joint (hip, knee, ankle) over the last swing, and its phase:
    when it touched down, as a share of its own step period after the other foot's touchdown
    (0.5 = exactly alternating). Updated once per control step, however many terms ask.
    """

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        robot = env.scene["robot"]
        self.joint_ids = torch.tensor(
            [
                [robot.find_joints(f"^{side}_{j}_joint$")[0][0] for j in STEP_PITCH_JOINTS]
                for side in sit.SIDES
            ],
            device=dev,
        )  # [2 legs, 3 joints]
        self.step = -1
        self.t = torch.zeros(n, device=dev)
        self.prev_down = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.force_sum = torch.zeros(n, 2, device=dev)
        self.force_n = torch.zeros(n, 2, device=dev)
        self.q_max = torch.full((n, 2, 3), -math.inf, device=dev)
        self.q_min = torch.full((n, 2, 3), math.inf, device=dev)
        self.lift_t = torch.zeros(n, 2, device=dev)
        self.touch_t = torch.full((n, 2), -math.inf, device=dev)
        self.air = torch.zeros(n, 2, device=dev)
        self.stance_force = torch.zeros(n, 2, device=dev)
        self.swing_range = torch.zeros(n, 2, 3, device=dev)
        self.phase = torch.full((n, 2), 0.5, device=dev)
        self.has_swing = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.has_stance = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.has_phase = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    @staticmethod
    def of(env, update: bool = True) -> StepTracker:
        tracker = getattr(env, "_laai_steps", None)
        if tracker is None:
            tracker = env._laai_steps = StepTracker(env)
        if update:
            tracker.update(env)
        return tracker

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.t[ids] = 0.0
        self.prev_down[ids] = True
        self.force_sum[ids] = self.force_n[ids] = 0.0
        self.q_max[ids], self.q_min[ids] = -math.inf, math.inf
        self.lift_t[ids] = 0.0
        self.touch_t[ids] = -math.inf
        self.air[ids] = self.stance_force[ids] = self.swing_range[ids] = 0.0
        self.phase[ids] = 0.5
        self.has_swing[ids] = self.has_stance[ids] = self.has_phase[ids] = False

    def update(self, env) -> None:
        if env.common_step_counter == self.step:
            return
        self.step = env.common_step_counter
        sensor = env.scene.sensors["feet_floor_contact"].data
        self.advance(
            env.step_dt,
            down=sensor.found > 0,
            force=sensor.force.norm(dim=-1) / sit.ROBOT_WEIGHT_N,
            q=env.scene["robot"].data.joint_pos[:, self.joint_ids],
        )

    def advance(self, dt: float, down, force, q) -> None:
        """One control step: feet down [worlds, 2], floor force per foot (share of the weight)
        [worlds, 2], pitch joints [worlds, 2 legs, 3]."""
        self.t += dt
        t = self.t.unsqueeze(-1).expand(-1, 2)
        lift, touch = self.prev_down & ~down, ~self.prev_down & down

        # Stance ends: keep its mean force.
        done = lift & (self.force_n > 0)
        self.stance_force = torch.where(
            done, self.force_sum / self.force_n.clamp(min=1.0), self.stance_force
        )
        self.has_stance |= done
        self.force_sum = torch.where(lift, 0.0, self.force_sum + force * down)
        self.force_n = torch.where(lift, 0.0, self.force_n + down.float())
        self.lift_t = torch.where(lift, t, self.lift_t)

        # Swing: track each pitch joint's range; on touchdown keep it and the air time.
        swing = (~down).unsqueeze(-1)
        self.q_max = torch.where(swing, torch.maximum(self.q_max, q), self.q_max)
        self.q_min = torch.where(swing, torch.minimum(self.q_min, q), self.q_min)
        landed = touch & torch.isfinite(self.q_max).all(-1)
        self.swing_range = torch.where(
            landed.unsqueeze(-1), self.q_max - self.q_min, self.swing_range
        )
        self.air = torch.where(landed, t - self.lift_t, self.air)
        self.has_swing |= landed
        self.q_max = torch.where(touch.unsqueeze(-1), -math.inf, self.q_max)
        self.q_min = torch.where(touch.unsqueeze(-1), math.inf, self.q_min)

        # Phase on touchdown: time since the other foot's touchdown over this foot's period.
        other = self.touch_t.flip(-1)
        period = t - self.touch_t
        timed = touch & torch.isfinite(self.touch_t) & torch.isfinite(other) & (period > 0)
        phase = ((t - other) / period.clamp(min=1e-6)).clamp(0.0, 1.0)
        self.phase = torch.where(timed, phase, self.phase)
        self.has_phase |= timed
        self.touch_t = torch.where(touch, t, self.touch_t)
        self.prev_down = down

    def asymmetry(self) -> dict[str, torch.Tensor]:
        """Left against right: relative air time and force differences, mean pitch-range
        difference (rad), mean distance of the phases from 0.5; and whether all are current."""
        air, force = self.air, self.stance_force
        recent = (self.t.unsqueeze(-1) - self.touch_t < STEP_STALE_S).all(-1)
        return {
            "air": (air[:, 0] - air[:, 1]).abs() / air.sum(-1).clamp(min=1e-6),
            "force": (force[:, 0] - force[:, 1]).abs() / force.sum(-1).clamp(min=1e-6),
            "swing": (self.swing_range[:, 0] - self.swing_range[:, 1]).abs().mean(-1),
            "phase": (self.phase - 0.5).abs().mean(-1),
            "valid": (self.has_swing & self.has_stance & self.has_phase).all(-1) & recent,
        }


class step_symmetry:
    """Left and right steps alike and half a step apart, while moving.

    exp(-(air/std_ratio)^2 - (force/std_ratio)^2 - (swing/std_rad)^2 - (phase/std_phase)^2)
    on the latest completed step of each foot (StepTracker); 0 until both feet have stepped
    recently, so standing still or shuffling one foot earns nothing.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        self.env = env

    def reset(self, env_ids=None) -> None:
        StepTracker.of(self.env, update=False).reset(env_ids)

    def __call__(self, env, std_ratio: float, std_rad: float, std_phase: float) -> torch.Tensor:
        a = StepTracker.of(env).asymmetry()
        score = torch.exp(
            -torch.square(a["air"] / std_ratio)
            - torch.square(a["force"] / std_ratio)
            - torch.square(a["swing"] / std_rad)
            - torch.square(a["phase"] / std_phase)
        )
        moving = speed_command(env).abs() > MOVING
        return score * (a["valid"] & moving).float()


class step_asymmetry_metric:
    """One component of StepTracker.asymmetry(), holding its latest value ('valid': share of
    time with a current step on both feet)."""

    def __init__(self, cfg: MetricsTermCfg, env):
        self.env = env
        self.key = cfg.params["key"]

    def reset(self, env_ids=None) -> None:
        StepTracker.of(self.env, update=False).reset(env_ids)

    def __call__(self, env, key: str) -> torch.Tensor:
        value = StepTracker.of(env).asymmetry()[key].float()
        return torch.rad2deg(value) if key == "swing" else value


# --- Metrics (identical for both tasks) --------------------------------------------------------
def speed_error(env) -> torch.Tensor:
    return (speed_command(env) - bike_speed(env)).abs()


def turn_rate_error(env) -> torch.Tensor:
    return (turn_command(env) - bike_yaw_rate(env)).abs()


def commanded_heading_error_deg(env) -> torch.Tensor:
    return torch.rad2deg(CommandedHeading.of(env).error(env).abs())


def foot_slip_speed(env) -> torch.Tensor:
    """Mean horizontal speed of the feet that are on the floor (m/s)."""
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(FOOT_SITES)
    speed = robot.data.site_lin_vel_w[:, ids, :2].norm(dim=-1)
    down = _feet_down(env).float()
    return (speed * down).sum(-1) / down.sum(-1).clamp(min=1.0)


def torque_ratio_max(env) -> torch.Tensor:
    return torque_ratio(env).max(-1).values


def torque_above_90(env) -> torch.Tensor:
    """1 while any motor is above 90% of its limit (the robot's stop: 90% for 0.5 s)."""
    return (torque_ratio_max(env) > 0.9).float()


def simple_rewards(control_dt: float) -> dict[str, RewardTermCfg]:
    """Only the task: commanded speed, straight line, stay seated, do not fall, smooth."""
    return {
        "track_bike_speed": RewardTermCfg(
            func=track_bike_speed, weight=2.0, params={"std": SPEED_STD, "tau": SPEED_FILTER_S}
        ),
        "on_line": RewardTermCfg(func=on_line, weight=1.0, params={"std_m": 0.3, "std_rad": 0.3}),
        "saddle_load": RewardTermCfg(func=saddle_load, weight=1.0, params={"target": 0.5}),
        # Five reward units per fall (the manager otherwise scales weights by control dt).
        "failure": RewardTermCfg(func=envs_mdp.is_terminated, weight=-5.0 / control_dt),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.2),
    }


def shaped_rewards(control_dt: float) -> dict[str, RewardTermCfg]:
    """The simple rewards plus terms for the gait, the posture and the motor load."""
    policy_joints = SceneEntityCfg("robot", joint_names=POLICY_JOINT_REGEX)
    arms = r".*(shoulder|elbow).*"
    moving = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.4,
        r".*ankle_roll.*": 0.15,
        arms: 0.3,
    }
    return {
        **simple_rewards(control_dt),
        # Gait: real steps, no sliding foot, always one foot down, calm feet when still.
        "feet_air_time": RewardTermCfg(
            func=velocity_mdp.feet_air_time,
            weight=0.5,
            params={
                "sensor_name": "feet_floor_contact",
                "command_name": COMMAND,
                "command_threshold": MOVING,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=velocity_mdp.feet_slip,
            weight=-0.25,
            params={
                "sensor_name": "feet_floor_contact",
                "command_name": COMMAND,
                "command_threshold": MOVING,
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            },
        ),
        "both_feet_off": RewardTermCfg(func=both_feet_off, weight=-1.0),
        "feet_on_floor": RewardTermCfg(func=feet_on_floor_when_still, weight=0.5),
        # Posture: strict around the seated pose when still, loose legs when moving.
        "posture": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=0.5,
            params={
                "asset_cfg": policy_joints,
                "command_name": COMMAND,
                "std_standing": {r".*(hip|knee|ankle).*": 0.15, arms: 0.3},
                "std_walking": moving,
                "std_running": moving,
                "walking_threshold": MOVING,
                "running_threshold": 1.5,
            },
        ),
        "trunk_orientation": RewardTermCfg(
            func=sit.trunk_seated_orientation, weight=1.0, params={"std": 0.2}
        ),
        "bike_upright": RewardTermCfg(
            func=free.bike_upright, weight=1.0, params={"std": math.radians(10.0)}
        ),
        # Motors: hands only rest on the bar, no oscillation, no leaning on the torque limit.
        "arm_effort": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-1e-3,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=sit.ARM_JOINTS)},
        ),
        "action_acc_l2": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.02),
        "torque_near_limit": RewardTermCfg(
            func=torque_near_limit, weight=-1.0, params={"soft": TORQUE_SOFT_LIMIT}
        ),
    }


def form_rewards(control_dt: float) -> dict[str, RewardTermCfg]:
    """The shaped rewards plus the riding form: trunk along the bike, legs, collisions and
    step symmetry."""
    return {
        **shaped_rewards(control_dt),
        "trunk_to_bike": RewardTermCfg(
            func=trunk_to_bike, weight=1.0, params={"std_yaw": 0.15, "std_pitch": 0.25}
        ),
        "leg_band": RewardTermCfg(
            func=leg_band,
            weight=-0.5,
            params={
                "knee_band": KNEE_OUT_BAND,
                "foot_band": FOOT_OUT_BAND,
                "scale": LEG_BAND_SCALE,
            },
        ),
        "bike_collision": RewardTermCfg(func=bike_collision, weight=-1.0),
        "step_symmetry": RewardTermCfg(
            func=step_symmetry,
            weight=0.5,
            params={"std_ratio": 0.2, "std_rad": 0.15, "std_phase": 0.15},
        ),
    }


def steer_rewards(control_dt: float) -> dict[str, RewardTermCfg]:
    """The Form rewards with the straight line swapped for the commanded turn rate."""
    rewards = form_rewards(control_dt)
    del rewards["on_line"]
    rewards["track_bike_turn_rate"] = RewardTermCfg(
        func=track_bike_turn_rate, weight=1.0, params={"std": TURN_STD, "tau": TURN_FILTER_S}
    )
    return rewards


def trunk_bike_yaw_deg(env) -> torch.Tensor:
    return torch.rad2deg(trunk_yaw_pitch_to_bike(env)[0].abs())


def trunk_bike_pitch_deg(env) -> torch.Tensor:
    return torch.rad2deg(trunk_yaw_pitch_to_bike(env)[1].abs())


def knee_out_cm(env) -> torch.Tensor:
    return 100.0 * legs_outward(env)[0].mean(-1)


def foot_out_cm(env) -> torch.Tensor:
    return 100.0 * legs_outward(env)[1].mean(-1)


def bike_collision_sensor() -> ContactSensorCfg:
    """Every bike geom except the saddle (its slices and the retired original box), against
    any part of the robot. Visual-only geoms are listed too but never make contacts."""
    return ContactSensorCfg(
        name="bike_collision",
        primary=ContactMatch(mode="geom", pattern=r".*", entity="bike", exclude=(r"^saddle",)),
        secondary=ContactMatch(mode="subtree", pattern="robot/trunk"),
        fields=("found", "force"),
        reduce="netforce",
    )


def push_env_cfg(
    shaped: bool = False, play: bool = False, form: bool = False, steer: bool = False
) -> ManagerBasedRlEnvCfg:
    """The push task; `steer` (the Steer task) adds the turn rate to the command and the Form
    rewards, swapping the straight line for it (`shaped` and `form` are then implied)."""
    # The free bike, its fall limits (bike 20 deg, trunk 35 deg) and the critic's bike state.
    cfg = free.free_sit_env_cfg(play=play)
    # Air time per foot, for the stepping reward and the step count.
    cfg.scene.sensors = tuple(
        dataclasses.replace(s, track_air_time=True) if s.name == "feet_floor_contact" else s
        for s in cfg.scene.sensors
    ) + (bike_collision_sensor(),)
    if not play:
        cfg.episode_length_s = 20.0  # a few speed changes per episode

    stages = STEER_STAGES if steer else [{**s, "ang_vel_z": (0.0, 0.0)} for s in SPEED_STAGES]
    stage = stages[-1] if play else stages[0]
    command_cls = BikeTurnCommandCfg if steer else UniformVelocityCommandCfg
    cfg.commands = {
        COMMAND: command_cls(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=STANDING_SHARE,
            heading_command=False,
            rel_heading_envs=0.0,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=stage["lin_vel_x"], lin_vel_y=(0.0, 0.0), ang_vel_z=stage["ang_vel_z"]
            ),
        )
    }
    command_obs = ObservationTermCfg(
        func=envs_mdp.generated_commands, params={"command_name": COMMAND}
    )
    cfg.observations["actor"].terms["command"] = command_obs
    cfg.observations["critic"].terms["command"] = dataclasses.replace(command_obs)
    if steer:
        cfg.observations["critic"].terms["bike_heading"] = ObservationTermCfg(func=bike_heading_obs)
    else:
        cfg.observations["critic"].terms["bike_line"] = ObservationTermCfg(func=bike_line_obs)

    if not play:
        # One shift for the whole saddle (shared_random), new in every episode.
        cfg.events["saddle_position"] = EventTermCfg(
            mode="reset",
            func=dr.geom_pos,
            params={
                "ranges": {
                    0: (-SADDLE_SHIFT_X, SADDLE_SHIFT_X),
                    2: (-SADDLE_SHIFT_Z, SADDLE_SHIFT_Z),
                },
                "operation": "add",
                "shared_random": True,
                "asset_cfg": SceneEntityCfg("bike", geom_names=(r"saddle_surface_\d+",)),
            },
        )
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    if steer:
        cfg.rewards = steer_rewards(control_dt)
    elif form:
        cfg.rewards = form_rewards(control_dt)
    else:
        cfg.rewards = shaped_rewards(control_dt) if shaped else simple_rewards(control_dt)
    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=velocity_mdp.commands_vel,
            params={
                "command_name": COMMAND,
                "velocity_stages": STEER_STAGES if steer else SPEED_STAGES,
            },
        )
    if steer:  # no straight line: heading against the commanded one instead
        course = {"commanded_heading_error_deg": MetricsTermCfg(func=commanded_heading_error_deg)}
    else:
        course = {
            "bike_distance_m": MetricsTermCfg(func=bike_distance, reduce="last"),
            "lateral_deviation_m": MetricsTermCfg(func=lateral_deviation),
            "heading_error_deg": MetricsTermCfg(func=heading_error_deg),
        }
    cfg.metrics.update(
        {
            "bike_speed_m_s": MetricsTermCfg(func=bike_speed),
            "speed_error_m_s": MetricsTermCfg(func=speed_error),
            "bike_yaw_rate_rad_s": MetricsTermCfg(func=bike_yaw_rate),
            "turn_rate_error_rad_s": MetricsTermCfg(func=turn_rate_error),
            **course,
            "both_feet_off": MetricsTermCfg(func=both_feet_off),
            "foot_slip_m_s": MetricsTermCfg(func=foot_slip_speed),
            "torque_ratio_max": MetricsTermCfg(func=torque_ratio_max, reduce="max"),
            "torque_above_90": MetricsTermCfg(func=torque_above_90),
            # Riding form (all tasks, so the Form task can be compared with the others).
            "trunk_bike_yaw_deg": MetricsTermCfg(func=trunk_bike_yaw_deg),
            "trunk_bike_pitch_deg": MetricsTermCfg(func=trunk_bike_pitch_deg),
            "knee_out_cm": MetricsTermCfg(func=knee_out_cm),
            "foot_out_cm": MetricsTermCfg(func=foot_out_cm),
            "bike_collision": MetricsTermCfg(func=bike_collision_any),
            "bike_collision_force": MetricsTermCfg(func=bike_collision_force),
            **{
                f"step_{key}_asymmetry": MetricsTermCfg(
                    func=step_asymmetry_metric, params={"key": key}
                )
                for key in ("air", "force", "phase")
            },
            "step_swing_asymmetry_deg": MetricsTermCfg(
                func=step_asymmetry_metric, params={"key": "swing"}
            ),
            "steps_tracked": MetricsTermCfg(func=step_asymmetry_metric, params={"key": "valid"}),
        }
    )
    return cfg


def push_runner_cfg():
    base = sit.k1_sit_ppo_runner_cfg()
    cfg = PolicyInitRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    cfg.experiment_name = "laai_push"
    cfg.max_iterations = 3000
    return cfg


for _task_id, _shaped, _form in (
    (SIMPLE_TASK_ID, False, False),
    (SHAPED_TASK_ID, True, False),
    (FORM_TASK_ID, True, True),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=push_env_cfg(shaped=_shaped, form=_form),
        play_env_cfg=push_env_cfg(shaped=_shaped, play=True, form=_form),
        rl_cfg=push_runner_cfg(),
        runner_cls=PushRunner,
    )
register_mjlab_task(
    task_id=STEER_TASK_ID,
    env_cfg=push_env_cfg(steer=True),
    play_env_cfg=push_env_cfg(play=True, steer=True),
    rl_cfg=push_runner_cfg(),
    runner_cls=PushSteerRunner,
)
