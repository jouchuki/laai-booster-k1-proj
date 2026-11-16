"""Checks on the phase 3 sit task (experiments/sit/task.py). Skipped when mjlab is not installed
(`uv sync --group train`); builds the scene and runs a short CPU rollout on Linux and macOS."""

import pytest

pytest.importorskip("mjlab")

import mujoco
import torch
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg

from experiments.sit import task as sit
from experiments.walk import task as walk


def test_sit_task_is_registered():
    assert sit.TASK_ID in list_tasks()


def test_actor_matches_the_walk_policy_and_booster_deploy():
    """Same 690-number input as the walk policy, so it deploys through booster_deploy the same
    way (with the seated pose as its default pose)."""
    sit_actor = load_env_cfg(sit.TASK_ID).observations["actor"]
    walk_actor = load_env_cfg(walk.TASK_ID).observations["actor"]
    assert list(sit_actor.terms) == list(walk_actor.terms)
    assert sit_actor.history_length == walk_actor.history_length == 10


def test_scene_holds_the_bike_and_welds_the_hands():
    model = Scene(sit.sit_scene_cfg(), "cpu").compile()
    bike_joints = [
        model.joint(j) for j in range(model.njnt) if model.joint(j).name.startswith("bike/")
    ]
    assert all(j.type != mujoco.mjtJoint.mjJNT_FREE for j in bike_joints), (
        "bike fixed (held upright)"
    )
    assert {j.name for j in bike_joints} == {"bike/steer", "bike/front_wheel", "bike/rear_wheel"}
    assert model.neq == 2 and all(model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD for i in range(2))
    assert [model.geom(g).name for g in range(model.ngeom) if model.geom_type[g] == 0] == ["floor"]


def test_holding_the_seated_pose_puts_most_weight_on_the_saddle():
    """Zero actions = the seated pose held by the PD gains: the starting state must already sit
    (the policy then learns to keep it so under pushes and model errors)."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = sit.k1_sit_env_cfg(play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    terminated = 0
    for _ in range(75):  # 1.5 s
        _, _, term, _, _ = env.step(zero)
        terminated += int(term.sum())
    assert terminated == 0
    assert float(sit.saddle_load_fraction(env).min()) > 0.6
    assert float(sit.feet_on_floor(env).min()) == 1.0
    assert float(torch.rad2deg(sit.trunk_tilt_from_seated(env)).max()) < 8.0
    env.close()


# --- Stage 1: the bike leans on a softening stand ---------------------------------------------
def test_stage1_bike_leans_on_a_spring_stand():
    model = Scene(sit.sit_scene_cfg(stage=1), "cpu").compile()
    stand = model.joint(f"bike/{sit.STAND_JOINT}")
    assert stand.type == mujoco.mjtJoint.mjJNT_HINGE
    assert list(model.jnt_axis[stand.id]) == [1.0, 0.0, 0.0], "lean about the tyres' floor line"
    assert model.jnt_stiffness[stand.id] == sit.STAND_STIFFNESS_MAX


def test_stages_share_inputs_so_checkpoints_carry_over():
    """Stage 1 starts from a stage 0 checkpoint (scripts/train.py --init-from): actor and
    critic inputs and the actions must be the same size, and both log into laai_sit."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_rl_cfg

    dims = []
    for stage in (0, 1):
        cfg = sit.k1_sit_env_cfg(play=True, stage=stage)
        env = ManagerBasedRlEnv(cfg, device="cpu")
        obs, _ = env.reset()
        dims.append(
            (obs["actor"].shape[-1], obs["critic"].shape[-1], env.action_manager.total_action_dim)
        )
        env.close()
    assert dims[0] == dims[1] == (690, dims[0][1], 20)
    assert load_rl_cfg(sit.TASK_ID).experiment_name == load_rl_cfg(sit.LEAN_TASK_ID).experiment_name


def test_lean_curriculum_softens_the_stand_and_pushes_harder():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = sit.k1_sit_env_cfg(stage=1)
    cfg.scene.num_envs = 5
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    term = env.curriculum_manager.get_term_cfg("lean")
    curriculum = term.func
    stand = curriculum.jid
    stiffness = env.sim.model.jnt_stiffness
    assert float(stiffness[:, stand].min()) == sit.STAND_STIFFNESS_MAX, "starts stiff"

    curriculum.level = 1.0  # as if the policy had passed every step
    curriculum(env, torch.arange(5), **term.params)
    assert float(stiffness[1, stand]) == 0.0, "no spring left"
    assert float(stiffness[0, stand]) == sit.STAND_STIFFNESS_ANCHOR, "one world in five keeps some"
    push = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]["y"]
    assert push == (-2.0, 2.0)

    env.episode_length_buf[:] = 100  # five finished episodes, none ran to the time limit
    env.termination_manager.time_outs[:] = False
    curriculum(env, torch.arange(5), **term.params)
    assert curriculum.level == pytest.approx(0.9), "failing episodes make it easier again"
    env.close()
