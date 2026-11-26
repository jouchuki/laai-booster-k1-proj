"""The push tasks: the free bike, a forward-speed command, and two reward sets."""

import math

import pytest

pytest.importorskip("mjlab")

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from experiments.push import task as push
from experiments.push.runner import PushRunner


def test_bike_is_free_with_steering_and_nothing_holds_it():
    model = Scene(push.push_env_cfg().scene, "cpu").compile()
    assert model.joint("bike/root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("bike/steer").type[0] == mujoco.mjtJoint.mjJNT_HINGE
    names = [model.joint(i).name for i in range(model.njnt)]
    assert "bike/stand_roll" not in names and "bike/rail_x" not in names
    # Both constraints attach the hands to the bike, never the bike to the world.
    assert model.neq == 2 and (model.eq_obj1id > 0).all() and (model.eq_obj2id > 0).all()


def test_tasks_differ_only_in_their_rewards():
    simple, shaped = push.push_env_cfg(), push.push_env_cfg("shaped")
    symmetric = push.push_env_cfg("simple_symmetric")
    saddle = push.push_env_cfg("simple_weight_on_saddle")
    assert set(simple.rewards) == {
        "track_bike_speed",
        "on_line",
        "saddle_load",
        "failure",
        "action_rate_l2",
    }
    assert set(simple.rewards) < set(shaped.rewards) and len(shaped.rewards) == 15
    assert set(symmetric.rewards) == set(simple.rewards) | {"foot_balance"}
    # Weight-On-Saddle: the simple rewards, only the saddle target raised from 50% to 75%.
    assert set(saddle.rewards) == set(simple.rewards)
    assert simple.rewards["saddle_load"].params == {"target": 0.5}
    assert saddle.rewards["saddle_load"].params == {"target": 0.75}
    for name, term in simple.rewards.items():
        if name != "saddle_load":
            assert saddle.rewards[name].params == term.params, name
    for other in (shaped, symmetric, saddle):
        for name, term in simple.rewards.items():
            assert other.rewards[name].weight == term.weight
        for part in ("terminations", "events", "metrics", "curriculum", "commands"):
            assert set(getattr(simple, part)) == set(getattr(other, part)), part
        actor = other.observations["actor"].terms
        assert simple.observations["actor"].terms.keys() == actor.keys()
    for task_id in push.TASK_IDS.values():
        assert load_rl_cfg(task_id).experiment_name == "laai_push"
        assert load_runner_cls(task_id) is PushRunner
    watching = push.push_env_cfg(play=True)
    assert not watching.curriculum and "saddle_position" not in watching.events


def test_command_is_forward_speed_only():
    command = push.push_env_cfg().commands[push.COMMAND]
    assert command.ranges.lin_vel_x == push.SPEED_STAGES[0]["lin_vel_x"]
    assert command.ranges.lin_vel_y == command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.rel_standing_envs == push.STANDING_SHARE and not command.heading_command


@pytest.fixture(scope="module")
def env():
    cfg = push.push_env_cfg("shaped")
    cfg.scene.num_envs = 2
    cfg.seed = 42
    cfg.events.pop("push_robot")
    cfg.events.pop("saddle_position")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_rollout_keeps_the_walk_interface_and_zero_actions_do_not_move_the_bike(env):
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 690)
    actions = torch.zeros(2, 20)
    for _ in range(100):
        obs, reward, terminated, _, _ = env.step(actions)
        assert torch.isfinite(reward).all() and torch.isfinite(obs["actor"]).all()
        assert not terminated.any()
    # The command reaches the actor: each term keeps its own 10-frame history, so the
    # command (after 2 x 3 x 10 IMU values) is at 60-89, newest last.
    newest = obs["actor"][:, 87:90]
    assert torch.allclose(newest[:, 0], push.speed_command(env))
    assert torch.allclose(newest[:, 1:], torch.zeros(2, 2))
    assert push.bike_speed(env).abs().max() < 0.05
    assert (push.torque_ratio(env) <= 1.0 + 1e-5).all()


def test_speed_reward_pays_for_the_commanded_speed_only(env):
    env.reset()
    term = env.reward_manager.get_term_cfg("track_bike_speed")
    command = env.command_manager.get_term(push.COMMAND)
    command.vel_command_b[:, 0] = 0.3
    rewards = {}
    for speed in (0.0, 0.3, 0.6):
        term.func.reset()
        term.func.filtered[:] = speed  # as after rolling at this speed for a while
        velocity = torch.zeros(2, 6)
        velocity[:, 0] = speed
        env.scene["bike"].write_root_link_velocity_to_sim(velocity)
        env.sim.forward()
        rewards[speed] = term.func(env, **term.params)
    assert torch.allclose(rewards[0.3], torch.ones(2), atol=1e-4)
    assert (rewards[0.0] < 0.05).all() and (rewards[0.6] < 0.05).all()
    env.reset()


def test_line_reward_falls_with_sideways_drift_and_turning(env):
    env.reset()
    term = env.reward_manager.get_term_cfg("on_line")
    assert torch.allclose(term.func(env, **term.params), torch.ones(2), atol=1e-3)
    pose = env.scene["bike"].data.root_link_pose_w.clone()
    pose[0, 1] += 0.3  # one world: 30 cm beside the line
    half = torch.tensor(0.15)  # the other: turned 0.3 rad away from it
    pose[1, 3:7] = torch.tensor([half.cos(), 0.0, 0.0, half.sin()])
    env.scene["bike"].write_root_link_pose_to_sim(pose)
    env.sim.forward()
    reward = term.func(env, **term.params)
    assert torch.allclose(reward, torch.full((2,), math.exp(-1.0)), atol=0.02)
    assert push.lateral_deviation(env)[0] == pytest.approx(0.3, abs=1e-3)
    assert push.heading_error_deg(env)[1] == pytest.approx(math.degrees(0.3), abs=0.1)
    env.reset()


def test_foot_balance_pays_for_equal_floor_time_only(env):
    env.reset()
    sensor = env.scene.sensors["feet_floor_contact"]
    assert sensor.primary_names == ["left_ankle_roll_link", "right_ankle_roll_link"]
    term = push.foot_balance(None, env)
    params = {"std": push.FOOT_BALANCE_STD, "tau": push.FOOT_SHARE_S}
    found = sensor.data.found

    def run(pattern, seconds):
        """Feed per-step (left, right) foot contacts, repeating `pattern`."""
        term.reset()
        steps = round(seconds / env.step_dt)
        for i in range(steps):
            found[:] = torch.tensor(pattern[i % len(pattern)], dtype=found.dtype)
            reward = term(env, **params)
        return reward

    one_foot = run([(0, 1)], 2.0)  # left held up for 2 s, right pushing
    assert (one_foot < 0.05).all()
    # Alternating at one step per second: each foot down for half a second in turn.
    half = round(0.5 / env.step_dt)
    alternating = run([(1, 0)] * half + [(0, 1)] * half, 6.0)
    assert (alternating > 0.8).all()
    both_down = run([(1, 1)], 2.0)
    assert torch.allclose(both_down, torch.ones(2))
    env.reset()


def test_saddle_moves_as_one_piece_within_its_range():
    cfg = push.push_env_cfg()
    cfg.scene.num_envs = 4
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        bike = env.scene["bike"]
        ids, _ = bike.find_geoms(r"saddle_surface_\d+")
        geoms = bike.indexing.geom_ids[ids]
        nominal = env.sim.get_default_field("geom_pos")[geoms]
        seen = []
        for _ in range(3):
            env.reset()
            shift = env.sim.model.geom_pos[:, geoms] - nominal
            assert torch.allclose(shift, shift[:, :1].expand_as(shift), atol=1e-6), "one piece"
            assert shift[..., 0].abs().max() <= push.SADDLE_SHIFT_X + 1e-6
            assert shift[..., 2].abs().max() <= push.SADDLE_SHIFT_Z + 1e-6
            assert shift[..., 1].abs().max() < 1e-6, "never sideways"
            seen.append(shift[:, 0].clone())
        assert not torch.allclose(seen[0], seen[1]), "new in every episode"
        assert seen[0][:, 0].unique().numel() > 1, "differs between worlds"
        actions = torch.zeros(4, 20)
        for _ in range(100):  # the robot settles onto the moved saddle without falling
            _, reward, terminated, _, _ = env.step(actions)
            assert torch.isfinite(reward).all() and not terminated.any()
    finally:
        env.close()


def test_evaluation_reports_each_speed_for_policy_and_zero_actions():
    from experiments.push.evaluate import evaluate_policy

    def zero(obs):
        return torch.zeros(obs.batch_size[0], 20)

    report = evaluate_policy(zero, "cpu", speeds=(0.0, 0.2), num_envs=2)
    assert [row["command_mps"] for row in report["policy"]] == [0.0, 0.2]
    assert report["policy"] == report["zero_action"], "a zero policy is the baseline itself"
    still, moving = report["zero_action"]
    assert still["success_fraction"] == moving["success_fraction"] == 1.0
    assert still["speed_error_mps"] < 0.02
    assert moving["speed_error_mps"] == pytest.approx(0.2, abs=0.02), "the baseline stays put"
    assert moving["torque_ratio_max"] < 0.9
    # Seated still, both feet stay down.
    assert still["left_foot_down_share"] == still["right_foot_down_share"] == 1.0
    assert still["foot_imbalance"] == 0.0
