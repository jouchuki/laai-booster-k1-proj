"""Phase 1: K1 velocity-tracking walk on flat ground, built on mjlab's velocity task.

Starts from mjlab's Unitree G1 flat-terrain setup and swaps in the K1: its model,
Booster's k1_walk PD gains and default pose, K1 body/joint names, and command ranges
sized for a 95 cm robot. Importing this module registers the tasks with mjlab.

Train (Linux + NVIDIA):  uv run python scripts/train.py K1Bike-Walk-Flat
"""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.spec_config import CollisionCfg

from k1bike.robots import k1

TASK_ID = "K1Bike-Walk-Flat"

##
# Robot
##

K1_ACTUATORS = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=patterns,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
    )
    for patterns, stiffness, damping, effort in k1.ACTUATOR_GROUPS.values()
)

# Every collision geom touches the terrain; no self-collision. Feet get full friction.
K1_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*",),
    contype=0,
    conaffinity=1,
    condim={r"^(left|right)_foot_collision$": 3, ".*": 1},
    priority={r"^(left|right)_foot_collision$": 1, ".*": 0},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
)

# Same rule mjlab uses for the G1: a full-scale action moves a joint to 1/4 of the
# angle where its PD controller would hit the effort limit. The head is not in the action.
K1_ACTION_SCALE = {
    pattern: 0.25 * effort / stiffness
    for group, (patterns, stiffness, _, effort) in k1.ACTUATOR_GROUPS.items()
    if group != "head"
    for pattern in patterns
}


def k1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, k1.STANDING_HEIGHT),
            joint_pos=dict(k1.DEFAULT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        collisions=(K1_COLLISION,),
        spec_fn=k1.k1_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=K1_ACTUATORS,
            soft_joint_pos_limit_factor=0.9,
        ),
    )


##
# Environment
##


def k1_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": k1_robot_cfg()}

    # Flat ground for phase 1 (as mjlab's G1 flat task).
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
    for group in ("actor", "critic"):
        del cfg.observations[group].terms["height_scan"]
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)
    cfg.sim.njmax = 300
    cfg.sim.nconmax = None
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

    # The real K1 cannot measure its body's linear velocity (only the IMU's angular
    # velocity and gravity direction), and Booster's k1_walk policy does not use it.
    # So the actor (the deployed policy) must not either; the critic still sees it.
    del cfg.observations["actor"].terms["base_lin_vel"]

    # Sensors on the K1's feet.
    for sensor in cfg.scene.sensors:
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in k1.FOOT_SITES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    cfg.scene.sensors += (
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

    # The policy moves 20 joints; the head holds its default pose (as in Booster's k1_walk).
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = (r"^(?!aahead_).*",)
    action.scale = K1_ACTION_SCALE

    # Commands sized for a 95 cm robot (Booster's k1_walk allows up to 1.6 m/s forward).
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-0.3, 0.6)
    twist.ranges.lin_vel_y = (-0.3, 0.3)
    twist.ranges.ang_vel_z = (-0.8, 0.8)
    twist.viz.z_offset = 0.75
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        {"step": 0, "lin_vel_x": (-0.3, 0.6), "ang_vel_z": (-0.8, 0.8)},
        {"step": 5000 * 24, "lin_vel_x": (-0.3, 1.0), "ang_vel_z": (-1.2, 1.2)},
        {"step": 10000 * 24, "lin_vel_x": (-0.3, 1.4), "ang_vel_z": (-1.6, 1.6)},
    ]

    # Rewards: K1 body, site and joint names; foot targets scaled to a smaller robot.
    cfg.rewards["upright"].params["asset_cfg"].body_names = (k1.BASE_BODY,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (k1.BASE_BODY,)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = k1.FOOT_SITES
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params["target_height"] = 0.06
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = _pose_std(scale=1.0)
    cfg.rewards["pose"].params["std_running"] = _pose_std(scale=1.6)
    cfg.rewards["pose"].params["running_threshold"] = 1.0

    # Domain randomization on K1 parts.
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = k1.FOOT_GEOMS
    cfg.events["base_com"].params["asset_cfg"].body_names = (k1.BASE_BODY,)

    cfg.viewer.body_name = k1.BASE_BODY
    cfg.viewer.distance = 2.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist.ranges.lin_vel_x = (-0.3, 1.2)
        twist.ranges.ang_vel_z = (-1.2, 1.2)

    return cfg


def _pose_std(scale: float) -> dict[str, float]:
    """How far each joint may move from the default pose before the pose reward drops.
    Loose for joints that drive the stride, tight for balance-critical ones (values follow
    mjlab's G1 reasoning)."""
    base = {
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*elbow.*": 0.15,
        r"aahead_.*": 0.05,
    }
    return {pattern: std * scale for pattern, std in base.items()}


##
# PPO
##


def k1_walk_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
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
        experiment_name="k1_walk",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=20_000,
    )


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=k1_walk_env_cfg(),
    play_env_cfg=k1_walk_env_cfg(play=True),
    rl_cfg=k1_walk_ppo_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
