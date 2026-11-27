"""The push tasks: the free bike, a forward-speed command (plus a turn rate in the Steer task),
and the reward sets."""

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


def test_both_tasks_differ_only_in_their_rewards():
    simple, shaped = push.push_env_cfg(), push.push_env_cfg(shaped=True)
    assert set(simple.rewards) == {
        "track_bike_speed",
        "on_line",
        "saddle_load",
        "failure",
        "action_rate_l2",
    }
    assert set(simple.rewards) < set(shaped.rewards) and len(shaped.rewards) == 15
    for name, term in simple.rewards.items():
        assert shaped.rewards[name].weight == term.weight
    for part in ("terminations", "events", "metrics", "curriculum", "commands"):
        assert set(getattr(simple, part)) == set(getattr(shaped, part)), part
    assert simple.observations["actor"].terms.keys() == shaped.observations["actor"].terms.keys()
    form = push.push_env_cfg(shaped=True, form=True)
    assert set(form.rewards) - set(shaped.rewards) == {
        "trunk_to_bike",
        "leg_band",
        "bike_collision",
        "step_symmetry",
    }
    for name, term in shaped.rewards.items():
        assert form.rewards[name].weight == term.weight
    for part in ("terminations", "events", "metrics", "curriculum", "commands"):
        assert set(getattr(form, part)) == set(getattr(shaped, part)), part
    for task_id in (push.SIMPLE_TASK_ID, push.SHAPED_TASK_ID, push.FORM_TASK_ID):
        assert load_rl_cfg(task_id).experiment_name == "laai_push"
        assert load_runner_cls(task_id) is PushRunner
    watching = push.push_env_cfg(play=True)
    assert not watching.curriculum and "saddle_position" not in watching.events


def test_steer_task_swaps_the_line_for_the_turn_rate():
    from experiments.push.runner import PushSteerRunner

    form, steer = push.push_env_cfg(shaped=True, form=True), push.push_env_cfg(steer=True)
    assert set(steer.rewards) == set(form.rewards) - {"on_line"} | {"track_bike_turn_rate"}
    for name, term in steer.rewards.items():
        if name in form.rewards:
            assert form.rewards[name].weight == term.weight, name
    critic = steer.observations["critic"].terms
    assert "bike_heading" in critic and "bike_line" not in critic
    assert steer.observations["actor"].terms.keys() == form.observations["actor"].terms.keys()
    assert "lateral_deviation_m" not in steer.metrics
    assert {"commanded_heading_error_deg", "turn_rate_error_rad_s"} <= set(steer.metrics)
    command = steer.commands[push.COMMAND]
    assert isinstance(command, push.BikeTurnCommandCfg)
    assert command.ranges.ang_vel_z == push.STEER_STAGES[0]["ang_vel_z"]
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    stages = steer.curriculum["command_vel"].params["velocity_stages"]
    assert stages == push.STEER_STAGES
    assert [s["lin_vel_x"] for s in stages] == [s["lin_vel_x"] for s in push.SPEED_STAGES]
    play = push.push_env_cfg(play=True, steer=True).commands[push.COMMAND]
    assert play.ranges.ang_vel_z == push.STEER_STAGES[-1]["ang_vel_z"]
    assert load_runner_cls(push.STEER_TASK_ID) is PushSteerRunner
    assert load_rl_cfg(push.STEER_TASK_ID).experiment_name == "laai_push"


def test_command_is_forward_speed_only():
    command = push.push_env_cfg().commands[push.COMMAND]
    assert command.ranges.lin_vel_x == push.SPEED_STAGES[0]["lin_vel_x"]
    assert command.ranges.lin_vel_y == command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.rel_standing_envs == push.STANDING_SHARE and not command.heading_command


@pytest.fixture(scope="module")
def env():
    cfg = push.push_env_cfg(shaped=True)
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

    report = evaluate_policy(zero, "cpu", speeds=(0.0, 0.2), turns=((0.2, 0.1),), num_envs=2)
    assert [row["command_mps"] for row in report["policy"]] == [0.0, 0.2, 0.2]
    assert [row["command_yaw_rate_rad_s"] for row in report["policy"]] == [0.0, 0.0, 0.1]
    assert report["policy"] == report["zero_action"], "a zero policy is the baseline itself"
    still, moving, turning = report["zero_action"]
    # Zero actions do not turn the bike either: it falls behind the commanded heading.
    assert turning["lateral_deviation_m"] is None and moving["lateral_deviation_m"] is not None
    assert abs(turning["mean_yaw_rate_rad_s"]) < 0.01
    assert turning["yaw_rate_error_rad_s"] == pytest.approx(0.1, abs=0.01)
    assert turning["heading_drift_deg"] == pytest.approx(math.degrees(0.1 * 10.0), abs=3.0)
    assert moving["heading_drift_deg"] < 3.0
    with pytest.raises(ValueError):
        evaluate_policy(zero, "cpu", speeds=(), turns=((0.2, 0.2),), num_envs=2)
    assert still["success_fraction"] == moving["success_fraction"] == 1.0
    assert still["speed_error_mps"] < 0.02
    assert moving["speed_error_mps"] == pytest.approx(0.2, abs=0.02), "the baseline stays put"
    assert moving["torque_ratio_max"] < 0.9


@pytest.fixture(scope="module")
def steer_env():
    cfg = push.push_env_cfg(steer=True)
    cfg.scene.num_envs = 64
    cfg.seed = 5
    cfg.events.pop("push_robot")
    cfg.events.pop("saddle_position")
    cfg.commands[push.COMMAND].ranges.ang_vel_z = (-1.0, 1.0)  # wider than the cap allows
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_turn_command_is_capped_by_the_speed(steer_env):
    env = steer_env
    env.reset()
    command = env.command_manager.get_term(push.COMMAND)
    command._resample_command(torch.arange(env.num_envs))
    speed, turn = command.vel_command_b[:, 0], command.vel_command_b[:, 2]
    assert (turn.abs() <= push.MAX_CURVATURE * speed.abs() + 1e-6).all()
    assert (turn > 0).any() and (turn < 0).any()
    assert command.vel_command_b[:, 1].abs().max() == 0.0


def test_turn_reward_and_commanded_heading(steer_env):
    env = steer_env
    env.reset()
    term = env.reward_manager.get_term_cfg("track_bike_turn_rate")
    command = env.command_manager.get_term(push.COMMAND)
    command.vel_command_b[:] = 0.0
    command.vel_command_b[:, 0] = 0.3
    command.vel_command_b[:, 2] = 0.15
    rewards = {}
    for rate in (0.0, 0.15, 0.3):
        term.func.reset()
        term.func.filtered[:] = rate  # as after turning at this rate for a while
        velocity = torch.zeros(env.num_envs, 6)
        velocity[:, 5] = rate
        env.scene["bike"].write_root_link_velocity_to_sim(velocity)
        env.sim.forward()
        rewards[rate] = term.func(env, **term.params)
    assert torch.allclose(rewards[0.15], torch.ones(env.num_envs), atol=1e-3)
    assert (rewards[0.0] < 0.15).all() and (rewards[0.3] < 0.15).all()
    # The commanded heading integrates the command; a bike that does not turn falls behind it.
    heading = push.CommandedHeading.of(env, update=False)
    heading.reset()
    for k in range(50):
        env.common_step_counter += 1  # one control step per update, without stepping physics
        push.CommandedHeading.of(env)
    assert torch.allclose(heading.target, torch.full((env.num_envs,), 50 * env.step_dt * 0.15))
    error = push.bike_heading_obs(env)[:, 0]
    start = env.scene["bike"].data.heading_w
    assert torch.allclose(error, start - heading.target, atol=1e-4)
    term.func.reset()
    assert (heading.target == 0).all()
    env.reset()


@pytest.fixture(scope="module")
def form_env():
    cfg = push.push_env_cfg(shaped=True, form=True)
    cfg.scene.num_envs = 2
    cfg.seed = 7
    cfg.events.pop("push_robot")
    cfg.events.pop("saddle_position")
    env = ManagerBasedRlEnv(cfg, device="cpu")
    yield env
    env.close()


def test_seated_form_costs_nothing_and_touches_only_saddle_and_bar(form_env):
    env = form_env
    env.reset()
    sensor = env.scene.sensors["bike_collision"]
    assert not any(name.startswith("saddle") for name in sensor.primary_names)
    assert {"seat_post_collision", "main_tube_front", "front_wheel"} <= set(sensor.primary_names)
    for _ in range(50):
        _, reward, terminated, _, _ = env.step(torch.zeros(2, 20))
        assert torch.isfinite(reward).all() and not terminated.any()
    assert (push.bike_collision_any(env) == 0).all()
    knee, foot = push.legs_outward(env)
    assert ((knee > 0.0) & (knee < 0.04)).all() and ((foot > 0.0) & (foot < 0.08)).all()
    for name in ("leg_band", "bike_collision"):
        term = env.reward_manager.get_term_cfg(name)
        assert (term.func(env, **term.params) == 0).all(), name
    term = env.reward_manager.get_term_cfg("trunk_to_bike")
    assert (term.func(env, **term.params) > 0.85).all()


def test_trunk_to_bike_falls_when_the_trunk_turns_away_from_the_bike(form_env):
    env = form_env
    env.reset()
    pose = env.scene["robot"].data.root_link_pose_w.clone()
    half = torch.tensor(0.15)  # world 0: trunk turned 0.3 rad about the vertical
    turn = torch.tensor([half.cos(), 0.0, 0.0, half.sin()])
    from mjlab.utils.lab_api.math import quat_mul

    pose[0, 3:7] = quat_mul(turn.unsqueeze(0), pose[0:1, 3:7])[0]
    env.scene["robot"].write_root_link_pose_to_sim(pose)
    env.sim.forward()
    yaw, _ = push.trunk_yaw_pitch_to_bike(env)
    assert yaw[0].abs() == pytest.approx(0.3, abs=0.03) and yaw[1].abs() < 0.01
    term = env.reward_manager.get_term_cfg("trunk_to_bike")
    reward = term.func(env, **term.params)
    assert reward[0] < 0.1 and reward[1] > 0.9
    env.reset()


def _walk(tracker, steps, dt, right_offset, right_force=0.3, right_range=0.4):
    """Feed a gait: each foot 0.4 s down then 0.2 s up; right shifted by `right_offset` s."""
    for k in range(steps):
        t = k * dt
        down = torch.tensor([[(t % 0.6) < 0.4, ((t - right_offset) % 0.6) < 0.4]])
        force = torch.tensor([[0.3, right_force]])
        swing_l = 0.4 * math.sin(math.pi * ((t - 0.4) % 0.6) / 0.2) if not down[0, 0] else 0.0
        phase_r = (t - right_offset - 0.4) % 0.6
        swing_r = right_range * math.sin(math.pi * phase_r / 0.2) if not down[0, 1] else 0.0
        q = torch.tensor([[[swing_l] * 3, [swing_r] * 3]])
        tracker.advance(dt, down=down, force=force, q=q)


def test_step_tracker_scores_alternating_steps_as_symmetric(form_env):
    def tracker():
        t = push.StepTracker(form_env)  # built for the fixture's 2 worlds; keep world 0
        for name, value in vars(t).items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (2,):
                setattr(t, name, value[:1].clone())
        return t

    even = tracker()
    _walk(even, 200, 0.02, right_offset=0.3)
    a = even.asymmetry()
    assert a["valid"].all()
    assert a["phase"].item() < 0.05 and a["air"].item() < 0.05 and a["force"].item() < 0.01
    assert a["swing"].item() < 0.02

    together = tracker()  # both feet step at once: same form, but no time offset
    _walk(together, 200, 0.02, right_offset=0.0)
    assert together.asymmetry()["phase"].item() > 0.4

    lopsided = tracker()  # right foot pushes harder and swings less
    _walk(lopsided, 200, 0.02, right_offset=0.3, right_force=0.6, right_range=0.1)
    b = lopsided.asymmetry()
    assert b["force"].item() > 0.3 and b["swing"].item() > 0.2
