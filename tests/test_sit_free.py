"""The unsupported bike must actually fall, and transfer must not resume old training."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import Scene
from mjlab.tasks.registry import load_rl_cfg

from experiments.sit import task as sit
from experiments.sit_free import task as free
from experiments.sit_free.runner import PolicyInitRunner, initialize_policy


def test_free_bike_has_no_stand_or_world_attachment():
    model = Scene(free.free_sit_env_cfg().scene, "cpu").compile()
    root = model.joint("bike/root")
    assert root.type == mujoco.mjtJoint.mjJNT_FREE
    dof = model.jnt_dofadr[root.id]
    assert (model.dof_damping[dof : dof + 6] == 0).all()
    assert "bike/stand_roll" not in [model.joint(i).name for i in range(model.njnt)]
    # Both remaining constraints attach hands to the bike, never the bike to the world.
    assert model.neq == 2
    assert (model.eq_obj1id > 0).all() and (model.eq_obj2id > 0).all()
    cfg = free.free_sit_env_cfg()
    assert "stand_stiffness" not in cfg.events and not cfg.curriculum
    assert load_rl_cfg(free.TASK_ID).experiment_name == "laai_sit_free"
    assert load_rl_cfg(sit.TASK_ID).experiment_name == "laai_sit"


def test_bike_alone_topples_under_gravity():
    """A physical rollout, not just a configuration assertion: no invisible world support."""
    spec = free.free_bike()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    model = spec.compile()
    data = mujoco.MjData(model)
    angle = torch.deg2rad(torch.tensor(10.0)).item()
    data.qpos[3:7] = [torch.cos(torch.tensor(angle / 2)), torch.sin(torch.tensor(angle / 2)), 0, 0]
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert data.xmat[model.body("frame").id].reshape(3, 3)[2, 2] < 0.7


@pytest.mark.parametrize("axis", [0, 1])
def test_fall_rewards_detect_roll_and_pitch_and_ignore_yaw(axis):
    angle = torch.deg2rad(torch.tensor([0.0, 10.0, 25.0, 90.0]))
    gravity = torch.zeros(4, 3)
    gravity[:, axis] = angle.sin()
    gravity[:, 2] = -angle.cos()
    data = SimpleNamespace(
        projected_gravity_b=gravity,
        root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 5.0]]).expand(4, -1),
    )
    # A purely sideways saddle impact must not count as vertical support.
    force = torch.tensor([[[1000.0, 0.0, 0.0]]]).expand(4, -1, -1)
    scene = {"bike": SimpleNamespace(data=data)}
    scene = type("Scene", (dict,), {})(scene)
    scene.sensors = {"saddle_contact": SimpleNamespace(data=SimpleNamespace(force=force))}
    env = SimpleNamespace(scene=scene)
    assert torch.allclose(free.bike_tilt_deg(env), torch.tensor([0.0, 10.0, 25.0, 90.0]), atol=1e-4)
    assert free.bike_fell(env, torch.deg2rad(torch.tensor(20.0))).tolist() == [
        False,
        False,
        True,
        True,
    ]
    upright = free.bike_upright(env, std=0.2)
    assert (upright[:-1] > upright[1:]).all()
    assert free.bike_tilt_rate_l2(env).sum() == 0, "yaw must not be a lean penalty"
    assert free.saddle_vertical_load_fraction(env).sum() == 0
    assert free.supported_saddle_load(env, target=0.6, std=0.2).sum() == 0


@pytest.fixture(scope="module")
def env():
    cfg = free.free_sit_env_cfg()
    cfg.scene.num_envs = 2
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_free_rollout_is_finite_and_reset_restores_both_bodies(env):
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 690)
    assert obs["critic"].shape[-1] == 84
    actions = torch.zeros(2, 20)
    for _ in range(75):
        obs, reward, _, _, _ = env.step(actions)
        assert torch.isfinite(reward).all() and torch.isfinite(obs["actor"]).all()
    for entity in ("robot", "bike"):
        state = env.scene[entity].data.root_link_pose_w.clone()
        state[:, 0] += 2.0
        env.scene[entity].write_root_link_pose_to_sim(state)
    env.reset()
    for entity in ("robot", "bike"):
        data = env.scene[entity].data
        assert torch.allclose(data.root_link_pose_w[:, :3], data.default_root_state[:, :3])


@pytest.mark.parametrize("axis,falls", [(0, True), (1, True), (2, False)])
def test_managers_terminate_for_actual_bike_tilt_and_apply_failure_cost(env, axis, falls):
    env.reset()
    pose = env.scene["bike"].data.root_link_pose_w.clone()
    half_angle = torch.deg2rad(torch.tensor(25.0)) / 2
    pose[:, 3:7] = 0
    pose[:, 3] = half_angle.cos()
    pose[:, 4 + axis] = half_angle.sin()
    env.scene["bike"].write_root_link_pose_to_sim(pose)
    env.sim.forward()
    env.termination_manager.compute()
    assert env.termination_manager.get_term("bike_fell").tolist() == [falls, falls]
    failure = env.reward_manager.get_term_cfg("failure")
    cost = failure.func(env) * failure.weight * env.step_dt
    assert torch.allclose(cost, torch.full((2,), -5.0 if falls else 0.0))
    env.reset()


def test_policy_transfer_keeps_fresh_critic_optimizer_and_counters(env, tmp_path):
    wrapped = RslRlVecEnvWrapper(env)
    cfg = asdict(free.free_sit_runner_cfg())
    runner = PolicyInitRunner(wrapped, cfg, device="cpu")
    checkpoint = runner.alg.save()
    # Deliberately incompatible old critic: it must not be loaded at all.
    checkpoint["critic_state_dict"] = {"old_critic": torch.zeros(1)}
    checkpoint["iter"] = 800
    checkpoint["infos"] = {"env_state": {"common_step_counter": 19200}}
    for key, value in checkpoint["actor_state_dict"].items():
        checkpoint["actor_state_dict"][key] = value.clone()
    actor_weights = next(k for k in checkpoint["actor_state_dict"] if k.endswith("weight"))
    checkpoint["actor_state_dict"][actor_weights].fill_(0.125)
    path = tmp_path / "model_800.pt"
    torch.save(checkpoint, path)
    critic = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
    counter = env.common_step_counter
    provenance = initialize_policy(runner, Path(path))
    assert torch.equal(
        runner.alg.get_policy().state_dict()[actor_weights],
        checkpoint["actor_state_dict"][actor_weights],
    )
    assert all(torch.equal(v, runner.alg.critic.state_dict()[k]) for k, v in critic.items())
    assert not runner.alg.optimizer.state
    assert runner.current_learning_iteration == 0
    assert env.common_step_counter == counter
    assert provenance["source_iteration"] == 800
    assert len(provenance["sha256"]) == 64


def test_push_task_is_the_free_task_plus_a_push_curriculum():
    cfg, base = free.free_push_env_cfg(), free.free_sit_env_cfg()
    assert set(cfg.rewards) == set(base.rewards)
    assert set(cfg.terminations) == set(base.terminations)
    assert set(cfg.curriculum) == {"push"} and not base.curriculum
    assert not free.free_push_env_cfg(play=True).curriculum, "watching: no curriculum"
    assert load_rl_cfg(free.PUSH_TASK_ID).experiment_name == "laai_sit_free"


def test_push_curriculum_pushes_harder_and_backs_off():
    cfg = free.free_push_env_cfg()
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        term = env.curriculum_manager.get_term_cfg("push")
        curriculum = term.func
        velocity = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]
        curriculum(env, torch.arange(4), **term.params)
        assert velocity["x"] == velocity["y"] == (-free.PUSH_MIN, free.PUSH_MIN), "starts light"

        env.episode_length_buf[:] = 500  # four episodes ran to the time limit
        env.termination_manager.time_outs[:] = True
        curriculum(env, torch.arange(4), **term.params)
        assert curriculum.level == pytest.approx(0.1)
        push = free.PUSH_MIN + 0.1 * (free.PUSH_MAX - free.PUSH_MIN)
        assert velocity["y"] == pytest.approx((-push, push))

        curriculum.level = 1.0
        env.termination_manager.time_outs[:] = False  # four failures: easier again
        curriculum(env, torch.arange(4), **term.params)
        assert curriculum.level == pytest.approx(0.9)
        assert velocity["roll"] == (-0.3, 0.3), "only the speed range grows"
    finally:
        env.close()


def test_push_sweep_reports_the_zero_action_baseline():
    from experiments.sit_free.evaluate import push_sweep

    def zero(obs):
        return torch.zeros(obs.batch_size[0], 20)

    sweep = push_sweep(zero, "cpu", pushes=(0.2, 2.0), num_envs=2)
    assert sweep["pushes_mps"] == [0.2, 2.0]
    # A zero-action policy is the baseline itself: it never beats it.
    assert sweep["success_fraction"] == sweep["zero_action_success_fraction"]
    assert sweep["beats_zero_action_at_mps"] == []
    assert sweep["zero_action_success_fraction"][0] == 1.0, "a light push needs no balance"
    assert sweep["zero_action_success_fraction"][1] == 0.0, "a 2 m/s push tips the bike"
