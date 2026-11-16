"""Phase 3: sit on the bike. mjlab training tasks, one per stage of the curriculum:

  * "LAAI-K1-Sit-Puky" (stage 0): the bike is fixed upright; learn to stay seated.
  * "LAAI-K1-Sit-Puky-Lean" (stage 1): the bike stands on a virtual stand that lets it lean
    on its tyres, held by a spring that a curriculum softens to nothing as the policy gets
    better; learn to keep a standing bike upright with the feet, like waiting at a traffic
    light. Start it from a stage 0 checkpoint (scripts/train.py --init-from).

Both stages share every input, output and the critic's inputs, so a checkpoint of one loads
into the other. Details below are for stage 0 unless stated.

The K1 starts seated on the Puky (models/k1_on_bike/puky_bike.json): pelvis in the saddle's
dip, hands welded to the grips, feet on the floor. The bike is held upright: it is fixed to
the world (steering and wheels still turn), as phase 3 of the plan says. The policy learns to
stay seated with most of its weight on the saddle, the rest on its feet, the hands only
resting on the bar, and to keep doing so when pushed and when the motors and body differ from
the model.

Same policy interface as the walk policy (experiments/walk/task.py) and booster_deploy's
k1_walk task, so it deploys the same way: 20 joints, target = seated pose + 0.25 * action,
k1_walk PD gains, 69 observations per frame with 10 frames of history. The velocity command
is always zero here (phase 4 will use it to push the bike along).

Train with scripts/train.py (commands: docs/commands.md); importing this module registers the
task.
"""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from experiments.walk.task import get_k1_robot_cfg, k1_ppo_runner_cfg
from k1bike.paths import BIKES, DEFAULT_BIKE
from k1bike.robots.k1 import (
    ACTION_SCALE,
    HISTORY,
    JOINT_VEL_SCALE,
    MASS_BODIES,
    POLICY_JOINT_REGEX,
    k1_spec,
)
from k1bike.scene import load_bike
from k1bike.seat import add_hand_clip_sites, clip_hands_to_bike, load_seated_pose

TASK_ID = "LAAI-K1-Sit-Puky"  # stage 0: bike fixed upright
LEAN_TASK_ID = "LAAI-K1-Sit-Puky-Lean"  # stage 1: bike leans on a softening spring
BIKE = DEFAULT_BIKE  # the Puky
BIKE_XML = BIKES[BIKE]
POSE = load_seated_pose(BIKE_XML)
ROBOT_WEIGHT_N = 19.7 * 9.81
ARM_JOINTS = (r".*shoulder.*", r".*elbow.*")
SIDES = ("left", "right")

# Stage 1 stand: a hinge along the line where both tyres touch the floor (the bike frame's x
# axis through its origin, on the floor under the rear axle), with a spring towards upright.
# Robot + bike tip over with about m*g*h = 23.4 kg * 9.81 * 0.45 m = 100 Nm per radian of lean,
# so 400 Nm/rad holds the bike almost rigidly and 0 leaves it all to the robot.
STAND_JOINT = "stand_roll"
STAND_STIFFNESS_MAX = 400.0  # Nm/rad
STAND_STIFFNESS_ANCHOR = 100.0  # Nm/rad: one world in five never gets softer than this
STAND_DAMPING = 10.0  # Nms/rad (always on: tyre and floor losses)
STAND_RANGE = math.radians(30.0)  # the bike lies on the floor beyond this
BIKE_FALL_DEG = 20.0  # stage 1 episode ends beyond this lean


# --- Scene: K1 + fixed bike + floor + hand welds ----------------------------------------------
def k1_with_hand_clips() -> mujoco.MjSpec:
    spec = k1_spec()
    add_hand_clip_sites(spec)
    return spec


def fixed_bike() -> mujoco.MjSpec:
    """Stage 0: the bike with its saddle fitted to the visible one, fixed to the world."""
    spec = load_bike(BIKE_XML)
    spec.delete(spec.joint("root"))
    return spec


def standing_bike() -> mujoco.MjSpec:
    """Stage 1: the bike on a virtual stand that only lets it lean on its tyres (no rolling or
    turning), held towards upright by a spring whose stiffness the curriculum lowers."""
    spec = load_bike(BIKE_XML)
    spec.delete(spec.joint("root"))
    spec.body("frame").add_joint(
        name=STAND_JOINT,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[0.0, 0.0, 0.0],
        axis=[1.0, 0.0, 0.0],
        stiffness=STAND_STIFFNESS_MAX,
        damping=STAND_DAMPING,
        range=[-STAND_RANGE, STAND_RANGE],
    )
    return spec


def seat_scene(spec: mujoco.MjSpec) -> None:
    """Scene hook: a floor, and the hand welds (k1bike/seat.py) between robot and bike."""
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.5, 0.55, 0.6, 1],
    )
    clip_hands_to_bike(spec, POSE, robot_prefix="robot/", bike_prefix="bike/", add_keyframe=False)


def robot_cfg() -> EntityCfg:
    """The walk task's K1 (actuators, gains, delays), starting in the seated pose."""
    cfg = get_k1_robot_cfg()
    cfg.spec_fn = k1_with_hand_clips
    joints = {f"^{n}$": 0.0 for n in ("aahead_yaw_joint", "aahead_pitch_joint")}
    joints.update({f"^{n}$": float(v) for n, v in POSE.joints.items()})
    cfg.init_state = EntityCfg.InitialStateCfg(
        pos=tuple(POSE.trunk_pos),  # the bike's frame sits at the world origin
        rot=tuple(POSE.trunk_quat),
        joint_pos=joints,
        joint_vel={".*": 0.0},
    )
    return cfg


def bike_cfg(stage: int = 0) -> EntityCfg:
    return EntityCfg(
        spec_fn=standing_bike if stage == 1 else fixed_bike,
        init_state=EntityCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    )


def sit_scene_cfg(num_envs: int = 1, stage: int = 0) -> SceneCfg:
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=0.0,  # the fixed bike is at the world origin in every environment
        terrain=None,
        entities={"robot": robot_cfg(), "bike": bike_cfg(stage)},
        sensors=(
            # Saddle load: each saddle slice against the K1's trunk (pelvis).
            ContactSensorCfg(
                name="saddle_contact",
                primary=ContactMatch(mode="geom", pattern=r"^saddle_surface_\d+$", entity="bike"),
                secondary=ContactMatch(mode="body", pattern=r"^trunk$", entity="robot"),
                fields=("found", "force"),
                reduce="netforce",
            ),
            ContactSensorCfg(
                name="feet_floor_contact",
                primary=ContactMatch(
                    mode="subtree",
                    pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                    entity="robot",
                ),
                secondary=ContactMatch(mode="geom", pattern="floor"),
                fields=("found", "force"),
                reduce="netforce",
            ),
        ),
        spec_fn=seat_scene,
    )


# --- Observation, reward and termination terms -------------------------------------------------
def _seated_gravity() -> torch.Tensor:
    """Gravity as the trunk feels it when seated on the upright bike (trunk frame)."""
    rot = torch.zeros(9, dtype=torch.float64).numpy()
    mujoco.mju_quat2Mat(rot, torch.tensor(POSE.trunk_quat, dtype=torch.float64).numpy())
    return torch.tensor(rot.reshape(3, 3).T @ [0.0, 0.0, -1.0], dtype=torch.float32)


SEATED_GRAVITY = _seated_gravity()


def zero_command(env) -> torch.Tensor:
    """The walk policy's velocity command slot, always zero while sitting (booster_deploy
    sends zero when nobody touches the joystick)."""
    return torch.zeros(env.num_envs, 3, device=env.device)


def trunk_tilt_from_seated(env) -> torch.Tensor:
    """Angle (rad) between the trunk's gravity direction now and when seated upright."""
    gravity = env.scene["robot"].data.projected_gravity_b
    cos = (gravity * SEATED_GRAVITY.to(gravity.device)).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


def trunk_seated_orientation(env, std: float) -> torch.Tensor:
    return torch.exp(-torch.square(trunk_tilt_from_seated(env) / std))


def saddle_load_fraction(env, sensor_name: str = "saddle_contact") -> torch.Tensor:
    """Share of the K1's weight on the saddle (sum of the net force on every saddle slice)."""
    force = env.scene.sensors[sensor_name].data.force  # [B, slices, 3], global frame
    return force.norm(dim=-1).sum(-1) / ROBOT_WEIGHT_N


def saddle_load(env, target: float) -> torch.Tensor:
    """1 when at least `target` of the weight rests on the saddle, less below it."""
    return (saddle_load_fraction(env) / target).clamp(max=1.0)


def feet_on_floor(env, sensor_name: str = "feet_floor_contact") -> torch.Tensor:
    found = env.scene.sensors[sensor_name].data.found
    return (found > 0).float().mean(-1)


def saddle_obs(env) -> torch.Tensor:
    return saddle_load_fraction(env).unsqueeze(-1)


def feet_obs(env) -> torch.Tensor:
    return (env.scene.sensors["feet_floor_contact"].data.found > 0).float()


def fell_off_seat(env, limit: float) -> torch.Tensor:
    return trunk_tilt_from_seated(env) > limit


# --- Stage 1: the bike's lean and the softening stand ----------------------------------------
def bike_lean(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Bike lean angle and rate (rad, rad/s) on the stand; zero for the fixed bike (stage 0)."""
    bike = env.scene["bike"]
    if STAND_JOINT not in bike.joint_names:
        zero = torch.zeros(env.num_envs, device=env.device)
        return zero, zero
    j = bike.joint_names.index(STAND_JOINT)
    return bike.data.joint_pos[:, j], bike.data.joint_vel[:, j]


def bike_lean_obs(env) -> torch.Tensor:
    """Critic only: the robot has no sensor on the bike (it feels the lean through its IMU)."""
    lean, rate = bike_lean(env)
    return torch.stack([lean, rate], dim=-1)


def bike_upright(env, std: float) -> torch.Tensor:
    lean, _ = bike_lean(env)
    return torch.exp(-torch.square(lean / std))


def bike_lean_rate_l2(env) -> torch.Tensor:
    _, rate = bike_lean(env)
    return torch.square(rate)


def bike_fell(env, limit: float) -> torch.Tensor:
    lean, _ = bike_lean(env)
    return lean.abs() > limit


def _stand_joint_id(env) -> int:
    bike = env.scene["bike"]
    return int(bike.indexing.joint_ids[bike.joint_names.index(STAND_JOINT)])


@requires_model_fields("jnt_stiffness")
def set_stand_stiffness(env, env_ids, stiffness: float) -> None:
    """Startup: one stand stiffness per world (also makes the field per-world, so the
    curriculum below can change it during training)."""
    env.sim.model.jnt_stiffness[:, _stand_joint_id(env)] = stiffness


class lean_curriculum:
    """Curriculum: soften the stand's spring and strengthen sideways pushes as the policy
    keeps the bike up.

    With both feet planted the seated K1 does not fall over even unsupported: holding the pose
    with zero actions, a 2.5 m/s sideways shove only sways the bike to 19 deg. So stage 1 is
    about staying upright (the reward pays for a lean within about 5 deg) against pushes that
    grow from `push_min` to `push_max` m/s (sideways speed given to the robot) as the stand
    softens.

    Counts finished episodes; once `batch` have finished (0: as many as there are environments), it moves one step softer if at least
    `promote` of them ran to the time limit (bike up, robot seated), one step stiffer below
    `demote`. Stiffness = max * (1 - level)^2, from STAND_STIFFNESS_MAX at level 0 to 0 at 1.
    One world in five never goes below STAND_STIFFNESS_ANCHOR, so the policy keeps practising
    the easier case (fine-tuning on the hardest case alone tends to forget it).
    """

    def __init__(self, cfg: CurriculumTermCfg, env):
        self.level = 0.0
        self.episodes = 0
        self.successes = 0
        self.last_rate = float("nan")
        self.jid = _stand_joint_id(env)
        self.anchor = torch.arange(env.num_envs, device=env.device) % 5 == 0

    def stiffness(self) -> float:
        return STAND_STIFFNESS_MAX * (1.0 - self.level) ** 2

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
        k = torch.full((env.num_envs,), self.stiffness(), device=env.device)
        k[self.anchor] = torch.clamp(k[self.anchor], min=STAND_STIFFNESS_ANCHOR)
        env.sim.model.jnt_stiffness[:, self.jid] = k
        push = push_min + self.level * (push_max - push_min)
        env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["y"] = (-push, push)
        return {
            "level": self.level,
            "stiffness": self.stiffness(),
            "push_mps": push,
            "success_rate": self.last_rate,
        }


# --- Environment -----------------------------------------------------------------------------
def k1_sit_env_cfg(play: bool = False, stage: int = 0) -> ManagerBasedRlEnvCfg:
    policy_joints = SceneEntityCfg("robot", joint_names=POLICY_JOINT_REGEX)
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.3, n_max=0.3),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "command": ObservationTermCfg(func=zero_command),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"biased": True, "asset_cfg": policy_joints},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": policy_joints},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=JOINT_VEL_SCALE,
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    critic_terms = {
        **{
            k: ObservationTermCfg(func=t.func, params=t.params, scale=t.scale)
            for k, t in actor_terms.items()
        },
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "saddle_load": ObservationTermCfg(func=saddle_obs),
        "feet_contact": ObservationTermCfg(func=feet_obs),
        # Zero in stage 0; present in both stages so their checkpoints load into each other.
        "bike_lean": ObservationTermCfg(func=bike_lean_obs),
    }
    cfg = ManagerBasedRlEnvCfg(
        decimation=4,
        scene=sit_scene_cfg(stage=stage),
        sim=SimulationCfg(
            njmax=400,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50),
        ),
        episode_length_s=10.0,
        viewer=ViewerConfig(body_name="trunk", distance=2.0),
        observations={
            # Exactly the walk policy's (and booster_deploy's) actor input.
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms=True,
                enable_corruption=not play,
                history_length=HISTORY,
                flatten_history_dim=True,
            ),
            "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True),
        },
        # The 20 policy joints, 0.25 around the seated pose (booster_deploy's action format).
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=POLICY_JOINT_REGEX,
                scale=ACTION_SCALE,
                use_default_offset=True,
            )
        },
        events={
            "reset_scene_to_default": EventTermCfg(
                func=envs_mdp.reset_scene_to_default, mode="reset"
            ),
            # Placed a little differently each time (small: the hands are welded to the bar).
            "reset_robot_joints": EventTermCfg(
                func=envs_mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.05, 0.05),
                    "velocity_range": (-0.2, 0.2),
                    "asset_cfg": policy_joints,
                },
            ),
            # The real motors and body are never exactly the model (as in walk run 2).
            "pd_gains": EventTermCfg(
                mode="startup",
                func=dr.pd_gains,
                params={
                    "kp_range": (0.85, 1.15),
                    "kd_range": (0.6, 1.6),
                    "operation": "scale",
                    "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                },
            ),
            "body_mass": EventTermCfg(
                mode="startup",
                func=dr.pseudo_inertia,
                params={
                    "alpha_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg("robot", body_names=MASS_BODIES),
                },
            ),
            "joint_friction": EventTermCfg(
                mode="startup",
                func=dr.joint_friction,
                params={"ranges": (0.0, 0.3), "operation": "abs", "asset_cfg": policy_joints},
            ),
            # Someone bumps the robot now and then; it must stay seated.
            "push_robot": EventTermCfg(
                func=envs_mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(2.0, 4.0),
                params={
                    "velocity_range": {
                        "x": (-0.2, 0.2),
                        "y": (-0.2, 0.2),
                        "roll": (-0.3, 0.3),
                        "pitch": (-0.3, 0.3),
                        "yaw": (-0.3, 0.3),
                    }
                },
            ),
        },
        rewards={
            # Sit like a rider: most weight on the saddle, feet on the floor, trunk as seated.
            "saddle_load": RewardTermCfg(func=saddle_load, weight=2.0, params={"target": 0.6}),
            "feet_on_floor": RewardTermCfg(func=feet_on_floor, weight=0.5),
            "trunk_orientation": RewardTermCfg(
                func=trunk_seated_orientation, weight=1.0, params={"std": 0.2}
            ),
            "posture": RewardTermCfg(
                func=envs_mdp.posture,
                weight=0.5,
                params={
                    "asset_cfg": policy_joints,
                    "std": {r".*(hip|knee|ankle).*": 0.15, r".*(shoulder|elbow).*": 0.3},
                },
            ),
            "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
            # Hands only rest on the bar: little arm effort against the welds.
            "arm_effort": RewardTermCfg(
                func=envs_mdp.joint_torques_l2,
                weight=-1e-3,
                params={"asset_cfg": SceneEntityCfg("robot", actuator_names=ARM_JOINTS)},
            ),
            # Calm, smooth motor commands (what keeps a real robot from shaking).
            "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.2),
            "action_acc_l2": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.02),
            "joint_acc_l2": RewardTermCfg(
                func=envs_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": policy_joints}
            ),
            "joint_torques_l2": RewardTermCfg(
                func=envs_mdp.joint_torques_l2,
                weight=-1e-5,
                params={"asset_cfg": SceneEntityCfg("robot", actuator_names=POLICY_JOINT_REGEX)},
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            # Trunk far from its seated orientation: slid off or slumped over.
            "fell_off_seat": TerminationTermCfg(
                func=fell_off_seat, params={"limit": math.radians(35.0)}
            ),
        },
    )
    if stage == 1:
        cfg.events["stand_stiffness"] = EventTermCfg(
            func=set_stand_stiffness,
            mode="startup",
            params={"stiffness": 0.0 if play else STAND_STIFFNESS_MAX},
        )
        # Keep the bike upright, without swinging it about.
        cfg.rewards["bike_upright"] = RewardTermCfg(
            func=bike_upright, weight=1.0, params={"std": math.radians(5.0)}
        )
        cfg.rewards["bike_lean_rate"] = RewardTermCfg(func=bike_lean_rate_l2, weight=-0.05)
        cfg.terminations["bike_fell"] = TerminationTermCfg(
            func=bike_fell, params={"limit": math.radians(BIKE_FALL_DEG)}
        )
        if not play:
            cfg.curriculum["lean"] = CurriculumTermCfg(
                func=lean_curriculum,
                params={
                    "step": 0.1,
                    "promote": 0.85,
                    "demote": 0.5,
                    "batch": 0,
                    "push_min": 0.3,
                    "push_max": 2.0,
                },
            )
    if play:
        # Watching: the finished case (stage 1: no spring), no pushes, no time limit.
        cfg.episode_length_s = int(1e9)
        cfg.events.pop("push_robot", None)
    return cfg


def k1_sit_ppo_runner_cfg():
    cfg = k1_ppo_runner_cfg()
    cfg.experiment_name = "laai_sit"
    cfg.max_iterations = 1500
    return cfg


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=k1_sit_env_cfg(),
    play_env_cfg=k1_sit_env_cfg(play=True),
    rl_cfg=k1_sit_ppo_runner_cfg(),
)
# Stage 1 logs into the same folder (logs/laai_sit/), so --init-from finds stage 0's runs.
register_mjlab_task(
    task_id=LEAN_TASK_ID,
    env_cfg=k1_sit_env_cfg(stage=1),
    play_env_cfg=k1_sit_env_cfg(play=True, stage=1),
    rl_cfg=k1_sit_ppo_runner_cfg(),
)
