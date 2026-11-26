"""The reference path, actor command and physical tracking stay aligned."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from experiments.joystick.trajectory import (
    MAX_SPEED_MPS,
    MAX_STEER_RAD,
    SEGMENT_SECONDS,
    WHEELBASE_M,
    sample_trajectories,
)


def test_generated_path_is_the_integral_of_the_commands():
    dt = 0.02
    path = sample_trajectories(16, dt, generator=torch.Generator().manual_seed(5))
    assert path.command.shape == (16, 1000, 2)
    assert path.pose.shape == (16, 1001, 3)
    assert path.stick.shape == path.command.shape
    assert (path.command[..., 0].abs() <= MAX_SPEED_MPS + 1e-6).all()
    assert (path.command[..., 1].abs() <= MAX_STEER_RAD + 1e-6).all()
    assert (path.command[..., 0] == 0).any()
    assert (path.command[..., 0] < 0).any() and (path.command[..., 0] > 0).any()
    assert (path.command[..., 1] > 0).any() and (path.command[..., 1] < 0).any()
    assert not torch.equal(path.pose[0], path.pose[1])

    speed, steer = path.command.unbind(-1)
    step = path.pose.diff(dim=1)
    yaw_delta = dt * speed * steer.tan() / WHEELBASE_M
    torch.testing.assert_close(step[..., 2], yaw_delta, atol=1e-6, rtol=0)
    midpoint = path.pose[:, :-1, 2] + yaw_delta / 2
    direction_xy = torch.stack((midpoint.cos(), midpoint.sin()), dim=-1)
    chord = dt * speed * torch.sinc(yaw_delta / (2 * math.pi))
    torch.testing.assert_close(step[..., :2], chord[..., None] * direction_xy, atol=1e-6, rtol=0)

    start = 0
    straight_reverse_count = 0
    for seconds in SEGMENT_SECONDS:
        end = start + round(seconds / dt)
        right, up = path.stick[:, start].unbind(-1)
        reverse = up < 0
        direction = torch.where(reverse, -1.0, 1.0)
        bearing = torch.atan2(-direction * right, direction * up)
        bearing = torch.where((right != 0) | (up != 0), bearing, 0.0)
        reached = path.pose[:, end, 2] - path.pose[:, start, 2]
        torch.testing.assert_close(reached, bearing, atol=2e-6, rtol=0)

        south = reverse & (right.abs() < 1e-6)
        straight_reverse_count += int(south.sum())
        torch.testing.assert_close(
            steer[south, start:end], torch.zeros_like(steer[south, start:end])
        )
        start_pose = path.pose[south, start]
        displacement = path.pose[south, end, :2] - start_pose[:, :2]
        heading = start_pose[:, 2]
        along_heading = displacement[:, 0] * heading.cos() + displacement[:, 1] * heading.sin()
        assert (along_heading < 0).all()
        start = end
    assert straight_reverse_count > 0


def test_joystick_task_steps_with_two_commands_and_scoped_reset():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from experiments.joystick.task import COMMAND, TASK_ID

    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.seed = 7
    cfg.scene.num_envs = 3
    assert set(cfg.rewards) == {
        "tracking",
        "saddle_load",
        "knee_height",
        "torque_near_limit",
        "failure",
    }
    assert cfg.rewards["saddle_load"].params["target"] == 0.78
    assert cfg.rewards["knee_height"].weight == -1.0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        term = env.command_manager.get_term(COMMAND)
        assert obs["actor"].shape == (3, 680)
        assert obs["critic"].shape == (3, 83)
        torch.testing.assert_close(term.command, term.feedforward[:, 0])
        for _ in range(4):
            _, reward, _, _, _ = env.step(torch.zeros(3, 20))
            assert torch.isfinite(reward).all()
        before = term.reference.clone()
        env.reset(env_ids=torch.tensor([1]))
        assert term.step_index.tolist() == [4, 0, 4]
        torch.testing.assert_close(term.reference[[0, 2]], before[[0, 2]])
        assert not torch.equal(term.reference[1], before[1])
        torch.testing.assert_close(term.command[1], term.feedforward[1, 0])

        # Moving the scheduled path to the bike's left should increase left steering.
        ids = torch.tensor([1])
        term.step_index[ids] = 1
        term.origin[ids, 1] += 0.1
        term.feedforward[ids, 1] = torch.tensor([0.3, 0.0])
        term._update_command(ids)
        assert term.command[1, 0] == pytest.approx(0.3)
        assert term.command[1, 1] > 0

        # A heading error of the same sign requires opposite steering in reverse.
        term.origin[ids, 1] -= 0.1
        term.reference[ids, 1, 2] += 0.1
        term.feedforward[ids, 1, 0] = -0.3
        term._update_command(ids)
        assert term.command[1, 0] == pytest.approx(-0.3)
        assert term.command[1, 1] < 0
    finally:
        env.close()


def test_shaped_joystick_uses_push_shaping_for_both_directions():
    from mjlab.envs import ManagerBasedRlEnv

    from experiments.joystick.task import COMMAND, joystick_env_cfg
    from experiments.push import task as push

    cfg = joystick_env_cfg(play=True, shaped=True)
    cfg.seed = 11
    cfg.scene.num_envs = 2
    expected = set(push.shaped_rewards(0.02)) - {"track_bike_speed", "on_line"}
    assert set(cfg.rewards) == expected | {"tracking", "knee_height"}
    assert cfg.rewards["saddle_load"].params["target"] == 0.78
    assert cfg.rewards["knee_height"].weight == -1.0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        path = env.command_manager.get_term(COMMAND)
        path.feedforward[:, :, 0] = torch.tensor([0.3, -0.3])[:, None]
        path._update_command()
        torch.testing.assert_close(
            env.command_manager.get_command(push.COMMAND),
            torch.tensor([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]),
        )
        assert obs["actor"].shape == (2, 680)
        for _ in range(3):
            _, reward, _, _, _ = env.step(torch.zeros(2, 20))
            assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_completion_rejects_an_idle_bike_on_an_out_and_back_track():
    from experiments.joystick.evaluate import completed_segments

    planned = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    progress = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.8, 0.0]])
    reached = torch.ones_like(planned, dtype=torch.bool)
    error = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    assert completed_segments(planned, progress, reached, error).all(dim=1).tolist() == [
        False,
        True,
    ]
