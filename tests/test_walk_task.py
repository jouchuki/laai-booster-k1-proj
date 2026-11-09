"""Checks that the mjlab walk task builds and steps. Skipped when mjlab is not installed
(`uv sync --group train`); runs on CPU, so it works on a Mac too."""

import pytest

pytest.importorskip("mjlab")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

from k1bike.tasks.walk import TASK_ID


def test_walk_task_is_registered():
    assert TASK_ID in list_tasks()


def test_policy_only_sees_what_the_real_robot_measures():
    actor_terms = load_env_cfg(TASK_ID).observations["actor"].terms
    assert "base_lin_vel" not in actor_terms, "the real K1 has no body-velocity sensor"


def test_walk_env_builds_and_steps():
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        action_dim = env.action_manager.total_action_dim
        assert action_dim == 20, "policy should control 20 joints (head excluded)"
        for _ in range(10):
            _obs, reward, *_ = env.step(torch.zeros(2, action_dim))
        assert torch.isfinite(reward).all()
    finally:
        env.close()
