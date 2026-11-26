"""Follow sampled joystick paths on the unsupported Puky."""

import math
from dataclasses import dataclass, fields

import torch
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task

from experiments.joystick.runner import JoystickRunner
from experiments.joystick.trajectory import (
    EPISODE_SECONDS,
    MAX_STEER_RAD,
    SEGMENT_SECONDS,
    sample_trajectories,
)
from experiments.push import task as push
from experiments.sit import task as sit
from experiments.sit_free.runner import PolicyInitRunnerCfg

TASK_ID = "LAAI-K1-Joystick-Puky"
SHAPED_TASK_ID = "LAAI-K1-Joystick-Puky-Shaped"
COMMAND = "joystick"
KNEE_LIMIT_M = 0.01


def wrap_angle(angle):
    return torch.atan2(angle.sin(), angle.cos())


@dataclass(kw_only=True)
class JoystickPathCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (EPISODE_SECONDS + 1, EPISODE_SECONDS + 1)

    def build(self, env):
        return JoystickPath(self, env)


class JoystickPath(CommandTerm):
    """Play a sampled reference and correct deviations through the steering target."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.steps = sum(round(seconds / env.step_dt) for seconds in SEGMENT_SECONDS)
        self.reference = torch.zeros(self.num_envs, self.steps + 1, 3, device=self.device)
        self.feedforward = torch.zeros(self.num_envs, self.steps, 2, device=self.device)
        self.world_ids = torch.arange(self.num_envs, device=self.device)
        self.step_index = torch.zeros_like(self.world_ids)
        self.origin = torch.zeros(self.num_envs, 3, device=self.device)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.bike = env.scene["bike"]
        [self.steer_id], _ = self.bike.find_joints("steer")

    @property
    def command(self):
        return self._command

    @property
    def steer_angle(self):
        return self.bike.data.joint_pos[:, self.steer_id]

    def compute(self, dt, env_ids=None):
        # Auto-reset worlds receive zero dt. Scoped reset calls also leave other worlds alone.
        if env_ids is None:
            self.step_index += dt > 0
        else:
            self.step_index[env_ids] += dt > 0
        super().compute(dt, env_ids)

    def _resample_command(self, env_ids):
        sample = sample_trajectories(len(env_ids), self._env.step_dt, device=self.device)
        self.reference[env_ids] = sample.pose
        self.feedforward[env_ids] = sample.command
        self.step_index[env_ids] = 0

    def path_errors(self, ids=None):
        ids = self.world_ids if ids is None else ids
        index = self.step_index[ids].clamp(max=self.steps)
        ref = self.reference[ids, index]
        origin = self.origin[ids]
        yaw = origin[:, 2]
        x, y = self.bike.data.root_link_pos_w[ids, :2].unbind(dim=-1)
        dx, dy = x - origin[:, 0], y - origin[:, 1]
        local_x = dx * yaw.cos() + dy * yaw.sin()
        local_y = -dx * yaw.sin() + dy * yaw.cos()
        ex, ey = ref[:, 0] - local_x, ref[:, 1] - local_y
        cross_track = -ex * ref[:, 2].sin() + ey * ref[:, 2].cos()
        heading = wrap_angle(yaw + ref[:, 2] - self.bike.data.heading_w[ids])
        return cross_track, heading, torch.hypot(ex, ey)

    def _update_command(self, env_ids=None):
        ids = self.world_ids if env_ids is None else env_ids
        index = self.step_index[ids]
        # This runs after sim.forward(), so the reset pose is available here.
        pose = torch.cat(
            (self.bike.data.root_link_pos_w[ids, :2], self.bike.data.heading_w[ids, None]), dim=1
        )
        self.origin[ids] = torch.where((index == 0)[:, None], pose, self.origin[ids])

        speed, steer = self.feedforward[ids, index.clamp(max=self.steps - 1)].unbind(dim=-1)
        speed = torch.where(index < self.steps, speed, 0.0)
        cross_track, heading, _ = self.path_errors(ids)
        # A bounded correction keeps the next command tied to the path after a disturbance.
        correction = (cross_track + 0.65 * speed.sign() * heading).clamp(-0.20, 0.20)
        steer = (steer + correction).clamp(-MAX_STEER_RAD, MAX_STEER_RAD)
        self._command[ids] = torch.stack((speed, torch.where(speed != 0, steer, 0.0)), dim=-1)

    def _update_metrics(self):
        pass


@dataclass(kw_only=True)
class MotionRewardCommandCfg(CommandTermCfg):
    """Expose the signed speed in the velocity shape expected by the Shaped gait terms."""

    resampling_time_range: tuple[float, float] = (EPISODE_SECONDS + 1, EPISODE_SECONDS + 1)

    def build(self, env):
        return MotionRewardCommand(self, env)


class MotionRewardCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self):
        self._command[:, 0] = self._env.command_manager.get_command(COMMAND)[:, 0]
        return self._command

    def _resample_command(self, env_ids):
        pass

    def _update_command(self, env_ids=None):
        pass

    def _update_metrics(self):
        pass


def tracking(env):
    term = env.command_manager.get_term(COMMAND)
    speed, steer = term.command.unbind(dim=-1)
    speed_match = torch.exp(-((push.bike_speed(env) - speed) / 0.15).square())
    steer_match = torch.exp(-((term.steer_angle - steer) / math.radians(8)).square())
    return (speed_match + steer_match) / 2


def knee_too_high(env, left, right):
    robot = env.scene["robot"]
    anchors = env.sim.data.xanchor
    violations = []
    for leg in (left, right):
        joint_ids = robot.indexing.joint_ids[leg.joint_ids]
        hip_z, knee_z = anchors[:, joint_ids, 2].unbind(dim=-1)
        violations.append(knee_z > hip_z + KNEE_LIMIT_M)
    return (violations[0] | violations[1]).float()


def path_error(env):
    term = env.command_manager.get_term(COMMAND)
    return term.path_errors()[2]


def joystick_env_cfg(play: bool = False, *, shaped: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = push.push_env_cfg(shaped=shaped, play=play)
    cfg.episode_length_s = EPISODE_SECONDS
    cfg.commands = {COMMAND: JoystickPathCfg()}
    if shaped:
        cfg.commands[push.COMMAND] = MotionRewardCommandCfg()
    cfg.curriculum.pop("command_vel", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": COMMAND}
        )
    cfg.observations["critic"].terms.pop("bike_line")
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    rewards = cfg.rewards if shaped else {}
    rewards.pop("track_bike_speed", None)
    rewards.pop("on_line", None)
    rewards.update(
        {
            "tracking": RewardTermCfg(func=tracking, weight=2.0),
            "saddle_load": RewardTermCfg(
                func=push.saddle_load, weight=1.0, params={"target": 0.78}
            ),
            "knee_height": RewardTermCfg(
                func=knee_too_high,
                weight=-1.0,
                params={
                    side: SceneEntityCfg(
                        "robot",
                        joint_names=(f"{side}_hip_pitch_joint", f"{side}_knee_pitch_joint"),
                        preserve_order=True,
                    )
                    for side in ("left", "right")
                },
            ),
            "torque_near_limit": RewardTermCfg(
                func=push.torque_near_limit, weight=-1.0, params={"soft": push.TORQUE_SOFT_LIMIT}
            ),
            "failure": push.simple_rewards(control_dt)["failure"],
        }
    )
    cfg.rewards = rewards
    for name in ("speed_error_m_s", "bike_distance_m", "lateral_deviation_m", "heading_error_deg"):
        cfg.metrics.pop(name)
    cfg.metrics["path_error_m"] = MetricsTermCfg(func=path_error)
    return cfg


def joystick_runner_cfg(shaped: bool) -> PolicyInitRunnerCfg:
    base = sit.k1_sit_ppo_runner_cfg()
    cfg = PolicyInitRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    cfg.experiment_name = "laai_joystick"
    if shaped:
        cfg.max_iterations = 5000
    return cfg


for _task_id, _shaped in ((TASK_ID, False), (SHAPED_TASK_ID, True)):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=joystick_env_cfg(shaped=_shaped),
        play_env_cfg=joystick_env_cfg(play=True, shaped=_shaped),
        rl_cfg=joystick_runner_cfg(_shaped),
        runner_cls=JoystickRunner,
    )
