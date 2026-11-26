"""Walk the bike along a straight line at a commanded speed, on the unsupported bike.

The K1 sits on the Puky as in the free-bike task (experiments/sit_free/task.py): seated pose,
hands welded to the grips, walk-policy interface (690 inputs, 20 joint actions). The bike is
free: nothing holds it upright or on course, and its steering turns under the robot's hands.
The velocity command (always zero while sitting) now carries a forward speed. The robot is
rewarded for pushing the bike along at that speed, on the line it started on. The saddle's
height and fore-aft position differ a little in every episode.

Two tasks share everything except their rewards, to compare a minimal reward with a shaped one:

  * LAAI-K1-Push-Puky-Simple: only the task (speed, straight line, stay seated, do not fall,
    smooth actions).
  * LAAI-K1-Push-Puky-Shaped: the same plus terms for the gait, posture and motor load.

Their returns are not comparable (different reward sums); compare the logged metrics and the
final evaluation (experiments/push/evaluate.py), which are identical for both.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import fields

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
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from experiments.push.runner import PushRunner
from experiments.sit import task as sit
from experiments.sit_free import task as free
from experiments.sit_free.runner import PolicyInitRunnerCfg
from experiments.sit_free.task import saddle_vertical_load_fraction
from k1bike.robots.k1 import DEPLOY_JOINTS, EFFORT, FOOT_SITES, POLICY_JOINT_REGEX

SIMPLE_TASK_ID = "LAAI-K1-Push-Puky-Simple"
SHAPED_TASK_ID = "LAAI-K1-Push-Puky-Shaped"
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


# --- Metrics (identical for both tasks) --------------------------------------------------------
def speed_error(env) -> torch.Tensor:
    return (speed_command(env) - bike_speed(env)).abs()


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


def push_env_cfg(shaped: bool = False, play: bool = False) -> ManagerBasedRlEnvCfg:
    # The free bike, its fall limits (bike 20 deg, trunk 35 deg) and the critic's bike state.
    cfg = free.free_sit_env_cfg(play=play)
    # Air time per foot, for the stepping reward and the step count.
    cfg.scene.sensors = tuple(
        dataclasses.replace(s, track_air_time=True) if s.name == "feet_floor_contact" else s
        for s in cfg.scene.sensors
    )
    if not play:
        cfg.episode_length_s = 20.0  # a few speed changes per episode

    final_range = SPEED_STAGES[-1]["lin_vel_x"]
    cfg.commands = {
        COMMAND: UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=STANDING_SHARE,
            heading_command=False,
            rel_heading_envs=0.0,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=final_range if play else SPEED_STAGES[0]["lin_vel_x"],
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
    }
    command_obs = ObservationTermCfg(
        func=envs_mdp.generated_commands, params={"command_name": COMMAND}
    )
    cfg.observations["actor"].terms["command"] = command_obs
    cfg.observations["critic"].terms["command"] = dataclasses.replace(command_obs)
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
    cfg.rewards = shaped_rewards(control_dt) if shaped else simple_rewards(control_dt)
    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=velocity_mdp.commands_vel,
            params={"command_name": COMMAND, "velocity_stages": SPEED_STAGES},
        )
    cfg.metrics.update(
        {
            "bike_speed_m_s": MetricsTermCfg(func=bike_speed),
            "speed_error_m_s": MetricsTermCfg(func=speed_error),
            "bike_distance_m": MetricsTermCfg(func=bike_distance, reduce="last"),
            "lateral_deviation_m": MetricsTermCfg(func=lateral_deviation),
            "heading_error_deg": MetricsTermCfg(func=heading_error_deg),
            "both_feet_off": MetricsTermCfg(func=both_feet_off),
            "foot_slip_m_s": MetricsTermCfg(func=foot_slip_speed),
            "torque_ratio_max": MetricsTermCfg(func=torque_ratio_max, reduce="max"),
            "torque_above_90": MetricsTermCfg(func=torque_above_90),
        }
    )
    return cfg


def push_runner_cfg():
    base = sit.k1_sit_ppo_runner_cfg()
    cfg = PolicyInitRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    cfg.experiment_name = "laai_push"
    cfg.max_iterations = 3000
    return cfg


for _task_id, _shaped in ((SIMPLE_TASK_ID, False), (SHAPED_TASK_ID, True)):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=push_env_cfg(shaped=_shaped),
        play_env_cfg=push_env_cfg(shaped=_shaped, play=True),
        rl_cfg=push_runner_cfg(),
        runner_cls=PushRunner,
    )
