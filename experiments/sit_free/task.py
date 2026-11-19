"""Sit and balance on the Puky with a free root: no world fixture, hinge or support spring.

The seated pose, motor model, actor interface and perturbations come from sit/task.py.
Hands remain welded to the grips, as in that task; this is still a model simplification.

Two tasks:
  * LAAI-K1-Sit-Puky-Free: pushes of at most 0.2 m/s, as sit/task.py (the archived comparison).
  * LAAI-K1-Sit-Puky-Free-Push: the same task with a push curriculum (push_curriculum), which
    raises the push speed from PUSH_MIN to PUSH_MAX as the policy keeps the bike up. At 0.2 m/s
    holding the seated pose with zero actions already passes every episode, so without stronger
    pushes there is no balance to learn.
"""

import math
from dataclasses import fields

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from experiments.sit import task as sit
from experiments.sit_free.runner import PolicyInitRunner, PolicyInitRunnerCfg
from k1bike.scene import load_bike

TASK_ID = "LAAI-K1-Sit-Puky-Free"
PUSH_TASK_ID = "LAAI-K1-Sit-Puky-Free-Push"
PUSH_MIN = 0.2  # m/s: sit/task.py's push, where the curriculum starts
PUSH_MAX = 2.0  # m/s: zero actions survive no episode from 1.0 m/s on (free-bike evaluation)


def free_bike():
    """Keep the model's original six-degree-of-freedom root and ordinary wheel damping."""
    return load_bike(sit.BIKE_XML)


def bike_tilt(env):
    """Total tilt from upright, detecting both roll and pitch, independently of heading."""
    gravity = env.scene["bike"].data.projected_gravity_b
    return torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))


def bike_state_obs(env):
    """Privileged critic inputs; the actor retains the original robot-only IMU interface."""
    data = env.scene["bike"].data
    return torch.cat(
        [data.projected_gravity_b, data.root_link_ang_vel_b, data.root_link_lin_vel_b], dim=-1
    )


def bike_upright(env, std):
    return torch.exp(-torch.square(bike_tilt(env) / std))


def bike_tilt_rate_l2(env):
    # Rotation around the bike's vertical axis is yaw, not a fall.
    return env.scene["bike"].data.root_link_ang_vel_b[:, :2].square().sum(-1)


def bike_fell(env, limit):
    return bike_tilt(env) > limit


def saddle_vertical_load_fraction(env):
    """Vertical support magnitude; lateral impacts cannot masquerade as weight support.

    Contact sensors report force on the saddle, so use its vertical magnitude rather than
    assuming the sign of the primary/secondary contact convention.
    """
    force = env.scene.sensors["saddle_contact"].data.force
    return force[..., 2].abs().sum(-1) / sit.ROBOT_WEIGHT_N


def supported_saddle_load(env, target, std):
    return (saddle_vertical_load_fraction(env) / target).clamp(max=1.0) * bike_upright(env, std)


def bike_tilt_deg(env):
    return torch.rad2deg(bike_tilt(env))


def bike_speed(env):
    return env.scene["bike"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)


def free_sit_env_cfg(play=False):
    cfg = sit.k1_sit_env_cfg(play=play)
    cfg.scene.entities["bike"] = EntityCfg(
        spec_fn=free_bike,
        init_state=EntityCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    )
    cfg.observations["critic"].terms["bike_lean"] = ObservationTermCfg(func=bike_state_obs)
    upright_std = math.radians(10.0)
    cfg.rewards["saddle_load"] = RewardTermCfg(
        func=supported_saddle_load,
        weight=2.0,
        params={"target": 0.6, "std": upright_std},
    )
    cfg.rewards["bike_upright"] = RewardTermCfg(
        func=bike_upright, weight=1.0, params={"std": upright_std}
    )
    cfg.rewards["bike_tilt_rate"] = RewardTermCfg(func=bike_tilt_rate_l2, weight=-0.05)
    # Five reward units per failure (the manager otherwise scales weights by control dt).
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    cfg.rewards["failure"] = RewardTermCfg(func=mdp.is_terminated, weight=-5.0 / control_dt)
    cfg.terminations["bike_fell"] = TerminationTermCfg(
        func=bike_fell, params={"limit": math.radians(sit.BIKE_FALL_DEG)}
    )
    cfg.metrics["saddle_load"] = MetricsTermCfg(func=saddle_vertical_load_fraction)
    cfg.metrics["bike_tilt_deg"] = MetricsTermCfg(func=bike_tilt_deg)
    cfg.metrics["bike_tilt_deg_max"] = MetricsTermCfg(func=bike_tilt_deg, reduce="max")
    cfg.metrics["bike_speed_m_s"] = MetricsTermCfg(func=bike_speed)
    return cfg


class push_curriculum:
    """Curriculum: push harder while the policy keeps the bike up and the robot seated.

    The push event (sit/task.py's push_robot, every 2-4 s) sets the robot's forward and
    sideways speed to a random value within +-push. Counts finished episodes; once `batch`
    have finished (0: as many as there are environments), the level goes up by `step` if at
    least `promote` of them ran to the time limit, down by `step` below `demote`.
    push = push_min + level * (push_max - push_min). Pushes are uniform within +-push, so light
    pushes stay part of training at every level.
    """

    def __init__(self, cfg: CurriculumTermCfg, env):
        self.level = 0.0
        self.episodes = 0
        self.successes = 0
        self.last_rate = float("nan")

    def __call__(
        self,
        env,
        env_ids,
        step: float,
        promote: float,
        demote: float,
        batch: int,
        push_min: float,
        push_max: float,
    ) -> dict[str, float]:
        ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.arange(env.num_envs, device=env.device)
        )
        ran = env.episode_length_buf[ids] > 0  # skip the reset at start-up
        self.episodes += int(ran.sum())
        self.successes += int((env.termination_manager.time_outs[ids] & ran).sum())
        if self.episodes >= (batch if batch > 0 else env.num_envs):
            self.last_rate = self.successes / self.episodes
            if self.last_rate >= promote:
                self.level = min(1.0, self.level + step)
            elif self.last_rate < demote:
                self.level = max(0.0, self.level - step)
            self.episodes = self.successes = 0
        push = push_min + self.level * (push_max - push_min)
        velocity = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]
        velocity["x"] = velocity["y"] = (-push, push)
        return {"level": self.level, "push_mps": push, "success_rate": self.last_rate}


def free_push_env_cfg(play=False):
    """The free-bike task with the push curriculum (watching: no pushes, as every sit task)."""
    cfg = free_sit_env_cfg(play=play)
    if not play:
        cfg.curriculum["push"] = CurriculumTermCfg(
            func=push_curriculum,
            params={
                "step": 0.1,
                "promote": 0.85,
                "demote": 0.5,
                "batch": 0,
                "push_min": PUSH_MIN,
                "push_max": PUSH_MAX,
            },
        )
    return cfg


def free_sit_runner_cfg():
    base = sit.k1_sit_ppo_runner_cfg()
    cfg = PolicyInitRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    cfg.experiment_name = "laai_sit_free"
    return cfg


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=free_sit_env_cfg(),
    play_env_cfg=free_sit_env_cfg(play=True),
    rl_cfg=free_sit_runner_cfg(),
    runner_cls=PolicyInitRunner,
)
# Logs into the same folder (logs/laai_sit_free/), so --agent.init-policy-from finds its runs.
register_mjlab_task(
    task_id=PUSH_TASK_ID,
    env_cfg=free_push_env_cfg(),
    play_env_cfg=free_push_env_cfg(play=True),
    rl_cfg=free_sit_runner_cfg(),
    runner_cls=PolicyInitRunner,
)
