"""Checks on the phase 1 walk task (phases/walk/task.py). Skipped when mjlab is not installed
(`uv sync --group train`); only builds the config, so it runs on Linux and macOS."""

import pytest

pytest.importorskip("mjlab")

from mjlab.tasks.registry import list_tasks, load_env_cfg

from phases.walk import task as walk


def test_walk_task_is_registered():
    assert walk.TASK_ID in list_tasks()


def test_deploy_settings_cover_all_joints():
    n = len(walk.DEPLOY_JOINTS)
    assert n == 22
    assert (
        len(walk.DEFAULT_POS) == len(walk.STIFFNESS) == len(walk.DAMPING) == len(walk.EFFORT) == n
    )
    assert len(walk.POLICY_JOINTS) == 20
    assert set(walk.POLICY_JOINTS) <= set(walk.DEPLOY_JOINTS)
    assert not any(j.startswith("aahead_") for j in walk.POLICY_JOINTS), "head is not controlled"


def test_actor_matches_booster_deploy_layout():
    actor = load_env_cfg(walk.TASK_ID).observations["actor"]
    assert list(actor.terms) == [
        "base_ang_vel",
        "projected_gravity",
        "command",
        "joint_pos",
        "joint_vel",
        "actions",
    ], "order must match booster_deploy's LocomotionPolicy.compute_observation"
    assert actor.history_length == walk.HISTORY == 10
