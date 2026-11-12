"""Checks on the phase 1 walk task (experiments/walk/task.py). Skipped when mjlab is not installed
(`uv sync --group train`); only builds the config, so it runs on Linux and macOS."""

import pytest

pytest.importorskip("mjlab")

from mjlab.tasks.registry import list_tasks, load_env_cfg

from experiments.walk import task as walk


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


def _collision_pairs(task_id):
    """(self-collision pairs, robot-ground pairs) in the compiled mjlab scene of a task."""
    import itertools

    from mjlab.scene import Scene

    model = Scene(load_env_cfg(task_id).scene, "cpu").compile()

    def is_robot(g):
        return model.body(model.geom_bodyid[g]).name.startswith("robot/")

    colliding = [
        g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    robot = [g for g in colliding if is_robot(g)]
    world = [g for g in colliding if not is_robot(g)]

    def can_touch(a, b):
        ba, bb = model.geom_bodyid[a], model.geom_bodyid[b]
        if ba == bb or model.body_parentid[ba] == bb or model.body_parentid[bb] == ba:
            return False
        return bool(
            model.geom_contype[a] & model.geom_conaffinity[b]
            or model.geom_contype[b] & model.geom_conaffinity[a]
        )

    self_pairs = sum(can_touch(a, b) for a, b in itertools.combinations(robot, 2))
    ground_pairs = sum(can_touch(a, b) for a in robot for b in world)
    return self_pairs, ground_pairs


def test_walk_task_keeps_self_collision():
    # The robot must not pass through itself (docs/decisions.md): Booster's model as is.
    self_pairs, ground_pairs = _collision_pairs(walk.TASK_ID)
    assert self_pairs == 174, "every K1 collision geom touches every non-adjacent one"
    assert ground_pairs == 20, "every robot collision geom touches the ground"
