"""Phase 1: Booster K1 velocity-tracking (walking) task for mjlab.

Validated on the real K1 (run 2, 2026-11-11): see README.md and runs/ in this folder.
Train with scripts/train.py, export with experiments/walk/export.py, deploy files in experiments/walk/deploy/
(commands: docs/commands.md).

The policy is set up to match Booster's own deploy tool (booster_deploy, task "k1_walk")
exactly, so a trained policy can run on the real K1 without changes:

  * controls the 20 body joints (head stays at its default), 50 Hz (0.005 s x 4)
  * target = default pose + 0.25 * action, PD gains/effort limits from booster_deploy
  * actor observation per frame (69): base angular velocity (3), projected gravity (3),
    velocity command (3), joint pos - default (20), 0.1 * joint vel (20), last action (20)
  * 10 frames of history (690 inputs)

Importing this module registers the task "LAAI-K1-Walk-Flat".

v2 (robot version): adds what the real motors do that the simulation didn't: 0-20 ms command
delay, +-15% stiffness / -40..+60% damping, +-10% body mass, joint friction and armature,
random start poses, more sensor noise, and stronger penalties on jittery actions.
"""

from __future__ import annotations

import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Robot model and booster_deploy settings, shared with the other phases. Re-exported here
# because export.py and the tests read them from this module.
from k1bike.robots.k1 import (  # noqa: F401
    ACTION_SCALE,
    BASE_HEIGHT,
    DAMPING,
    DEFAULT_POS,
    DELAY_MAX_STEPS,
    DEPLOY_JOINTS,
    EFFORT,
    FOOT_GEOMS,
    FOOT_SITES,
    HISTORY,
    JOINT_VEL_SCALE,
    MASS_BODIES,
    POLICY_JOINT_REGEX,
    POLICY_JOINTS,
    STIFFNESS,
    k1_spec,
)

TASK_ID = "LAAI-K1-Walk-Flat"


def get_k1_robot_cfg() -> EntityCfg:
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f"^{name}$",),
            stiffness=kp,
            damping=kd,
            effort_limit=eff,
            delay_min_lag=0,
            delay_max_lag=DELAY_MAX_STEPS,
        )
        for name, kp, kd, eff in zip(DEPLOY_JOINTS, STIFFNESS, DAMPING, EFFORT)
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT),
            joint_pos={f"^{n}$": float(p) for n, p in zip(DEPLOY_JOINTS, DEFAULT_POS)},
            joint_vel={".*": 0.0},
        ),
        spec_fn=k1_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=0.9
        ),
    )


def k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": get_k1_robot_cfg()}

    # Flat ground only.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)
    cfg.sim.njmax = 300
    cfg.sim.nconmax = None
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
    )

    # Actions: the 20 policy joints, scale 0.25 around the default pose (same as booster_deploy).
    cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=POLICY_JOINT_REGEX,
        scale=ACTION_SCALE,
        use_default_offset=True,
    )

    # Actor observations: exactly what booster_deploy computes on the robot.
    policy_joints = SceneEntityCfg("robot", joint_names=POLICY_JOINT_REGEX)
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "base_ang_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_ang_vel"},
                noise=Unoise(n_min=-0.3, n_max=0.3),
            ),
            "projected_gravity": ObservationTermCfg(
                func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
            ),
            "command": ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": "twist"}
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
                params={"biased": True, "asset_cfg": policy_joints},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": policy_joints},
                noise=Unoise(n_min=-1.5, n_max=1.5),
                scale=JOINT_VEL_SCALE,
            ),
            "actions": ObservationTermCfg(func=mdp.last_action),
        },
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=HISTORY,
        flatten_history_dim=True,
    )
    # Critic: privileged info (true velocity, contacts), no history, no terrain scan.
    critic = cfg.observations["critic"]
    critic.terms.pop("height_scan", None)

    # Commands sized for the K1 (booster_deploy clamps to +1.6/-0.3 m/s, 0.3 m/s side, 1.8 rad/s).
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-0.3, 0.6)
    twist.ranges.lin_vel_y = (-0.3, 0.3)
    twist.ranges.ang_vel_z = (-0.8, 0.8)
    twist.viz.z_offset = 0.8
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-0.3, 0.6), "ang_vel_z": (-0.8, 0.8)},
                {"step": 1500 * 24, "lin_vel_x": (-0.3, 1.0), "ang_vel_z": (-1.0, 1.0)},
            ],
        },
    )

    # Events / randomization (helps transfer to the real robot).
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.02)
    # Start from slightly different poses/speeds (e.g. when taking over from another controller).
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.5, 0.5)
    # The real motors and body are never exactly the model: randomize them per robot.
    cfg.events["pd_gains"] = EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "kp_range": (0.85, 1.15),
            "kd_range": (0.6, 1.6),
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        },
    )
    cfg.events["body_mass"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,  # mass and inertia together, about +-10%
        params={
            "alpha_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", body_names=MASS_BODIES),
        },
    )
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "ranges": (0.0, 0.3),
            "operation": "abs",
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_REGEX),
        },
    )
    cfg.events["joint_armature"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "ranges": (0.8, 1.2),
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_REGEX),
        },
    )
    push = cfg.events["push_robot"].params["velocity_range"]
    push.update(
        {
            "x": (-0.4, 0.4),
            "y": (-0.4, 0.4),
            "z": (-0.2, 0.2),
            "roll": (-0.3, 0.3),
            "pitch": (-0.3, 0.3),
            "yaw": (-0.5, 0.5),
        }
    )

    # Rewards: mjlab's velocity rewards, pointed at the K1's bodies/sites/joints.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params["target_height"] = 0.08
    # Smooth, calm motor commands (jittery commands are what make real robots shake).
    cfg.rewards["action_rate_l2"].weight = -0.2
    cfg.rewards["action_acc_l2"] = RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.02)
    # Small bonus for every step it stays up, so the early (large) smoothness penalties never
    # make "fall quickly to end the episode" look attractive.
    cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=1.0)
    cfg.rewards["joint_acc_l2"] = RewardTermCfg(
        func=envs_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_REGEX)},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=envs_mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=POLICY_JOINT_REGEX)},
    )
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=POLICY_JOINT_REGEX
    )
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*elbow.*": 0.15,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*elbow.*": 0.35,
    }
    # Same safety limit as booster_deploy (it stops the policy beyond ~60 degrees tilt).
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)

    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 2.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist.ranges.lin_vel_x = (-0.3, 1.0)
        twist.ranges.ang_vel_z = (-1.0, 1.0)
    return cfg


def k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="laai_walk",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=3_000,
    )


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=k1_flat_env_cfg(),
    play_env_cfg=k1_flat_env_cfg(play=True),
    rl_cfg=k1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
