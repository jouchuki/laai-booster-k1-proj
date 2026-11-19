"""The robot tasks "laai_sit_hold" and "laai_sit" (experiments/sit/deploy/laai_sit.py), rehearsed
in the K1 + bike scene. Needs booster_deploy (bash scripts/setup_booster.sh) and torch (uv sync
--group sim); skipped without them (CI clones only booster_assets)."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
REPO_ROOT = Path(__file__).resolve().parents[1]
if not any(
    (p / "booster_deploy").is_dir()
    for p in (REPO_ROOT / "booster_deploy", REPO_ROOT / "booster" / "booster_deploy")
):
    pytest.skip("booster_deploy not cloned", allow_module_level=True)

_spec = importlib.util.spec_from_file_location(
    "check_hold", REPO_ROOT / "experiments" / "sit" / "check_hold.py"
)
check_hold = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_hold)


@pytest.fixture(scope="module")
def task():
    return check_hold.load_task_module()


def test_task_holds_the_solved_puky_pose_and_never_walks(task):
    cfg = task.LaaiSitHoldTaskCfg()
    names = list(cfg.robot.joint_names)
    pose = json.loads((REPO_ROOT / "models" / "k1_on_bike" / "puky_bike.json").read_text())
    for name, angle in pose["joints"].items():
        assert cfg.robot.default_joint_pos[names.index(name)] == pytest.approx(angle)
    assert cfg.robot.default_joint_pos[:2] == [0.0, 0.0], "head straight"
    assert cfg.robot.prepare_mode == "standing", (
        "X slides to the pose; Booster's walking never runs"
    )
    assert list(cfg.robot.prepare_state.joint_pos) == list(cfg.robot.default_joint_pos)
    assert list(cfg.robot.prepare_state.stiffness) == list(cfg.robot.joint_stiffness)
    assert cfg.booster.exit_mode == "damping", "after any stop the robot goes limp, not walking"
    assert cfg.vel_command is None


def test_rehearsal_sits_with_most_weight_on_the_saddle():
    result = check_hold.rehearse(bike="puky_bike", seconds=4.0, offset=0.0)
    assert result["stopped"] is None
    assert result["split"]["saddle"] > 0.6
    assert result["split"]["hands"] < 0.15
    assert result["max_tilt_change_deg"] < 10


def test_rehearsal_survives_a_placement_error():
    result = check_hold.rehearse(bike="puky_bike", seconds=4.0, offset=0.1)
    assert result["stopped"] is None
    assert result["split"]["saddle"] > 0.5


def test_tilt_stop_trips_when_the_bike_falls(tmp_path, monkeypatch):
    """Holding a pose does not balance: a hard sideways push tips the bike over, and the task
    stops (damping) before the trunk is more than about 15 degrees off."""
    monkeypatch.chdir(tmp_path)  # the stop writes logs/stop_<time>.npz
    result = check_hold.rehearse(bike="puky_bike", seconds=6.0, offset=0.0, push=100.0)
    assert result["stopped"] and "tilted" in result["stopped"]
    assert result["max_tilt_change_deg"] < 20


# ---- laai_sit: the trained policy, with stand-in TorchScript policies --------------------------


class ZeroPolicy(torch.nn.Module):
    """Always zero actions: laai_sit must then behave exactly like laai_sit_hold."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(20) + 0.0 * x.reshape(-1)[:20]


class ShakingPolicy(torch.nn.Module):
    """Flips every action between +1 and -1 each step (the last action is the newest frame's
    last 20 inputs): the shaking a bad policy shows on the robot."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x.reshape(-1)[-20:]
        return torch.where(last > 0, -torch.ones(20), torch.ones(20))


def _save(module: torch.nn.Module, path: Path) -> Path:
    torch.jit.script(module).save(str(path))
    return path


def test_sit_task_runs_the_policy_around_the_seated_pose(task, tmp_path):
    hold = task.LaaiSitHoldTaskCfg()
    cfg = task.LaaiSitTaskCfg(policy_file=str(_save(ZeroPolicy(), tmp_path / "p.pt")))
    assert list(cfg.robot.default_joint_pos) == list(hold.robot.default_joint_pos), "seated pose"
    assert list(cfg.robot.prepare_state.joint_pos) == list(cfg.robot.default_joint_pos)
    assert cfg.robot.prepare_mode == "standing"
    assert cfg.booster.exit_mode == "damping"
    assert cfg.vel_command is None, "the policy always sees a zero velocity command"
    p = cfg.policy
    walk_joints = task.K1WalkControllerCfg().policy.policy_joint_names
    assert list(p.policy_joint_names) == list(walk_joints), "the export's joint order"
    assert (p.action_scale, p.actor_obs_history_length, p.obs_dof_vel_scale) == (0.25, 10, 0.1)
    assert p.history_init == "repeat" and p.action_filter == 1.0
    assert p.arm_action_fix_joint_name is None and not p.use_actor_module


def test_a_zero_policy_sits_exactly_like_the_hold(tmp_path):
    zero = _save(ZeroPolicy(), tmp_path / "zero.pt")
    sit = check_hold.rehearse(seconds=3.0, offset=0.0, task_name="laai_sit", policy=zero)
    hold = check_hold.rehearse(seconds=3.0, offset=0.0)
    assert sit["stopped"] is None and sit["task"] == "laai_sit"
    assert sit["split"]["saddle"] == pytest.approx(hold["split"]["saddle"], abs=1e-6)
    assert sit["max_tilt_change_deg"] == pytest.approx(hold["max_tilt_change_deg"], abs=1e-6)


def test_policy_actions_move_the_targets_by_a_quarter_in_their_own_joints(task, tmp_path):
    ctl = check_hold.make_controller(
        task, "puky_bike", "laai_sit", _save(ZeroPolicy(), tmp_path / "z.pt")
    )
    policy, names = ctl.policy, list(ctl.cfg.robot.joint_names)
    action = torch.zeros(20)
    action[policy.cfg.policy_joint_names.index("left_knee_pitch_joint")] = 1.0
    moved = policy._action_to_targets(action) - policy.default_joint_pos
    assert moved[names.index("left_knee_pitch_joint")] == pytest.approx(0.25)
    assert float(moved.abs().sum()) == pytest.approx(0.25)


def test_shaking_policy_is_stopped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the stop writes logs/stop_<time>.npz
    shaking = _save(ShakingPolicy(), tmp_path / "shaking.pt")
    result = check_hold.rehearse(
        seconds=2.0, offset=0.0, task_name="laai_sit", policy=shaking, bike_held=True
    )
    assert result["stopped"] and "shaking" in result["stopped"]
    assert result["held_s"] < 0.2
    stop = next((tmp_path / "logs").glob("stop_*.npz"))
    import numpy as np

    meta = json.loads(str(np.load(stop)["meta"]))
    assert meta["task"] == "laai_sit" and "shaking" in meta["reason"]


def test_ankle_kd15_task_only_raises_the_ankle_damping(task, tmp_path):
    from booster_deploy.utils.registry import get_task

    policy = str(_save(ZeroPolicy(), tmp_path / "p.pt"))
    base = task.LaaiSitTaskCfg(policy_file=policy)
    ankle = task.LaaiSitTaskCfg(policy_file=policy, ankle_kd=1.5)
    names = list(base.robot.joint_names)
    for i, name in enumerate(names):
        expected = 1.5 if name in task.ANKLE_JOINTS else base.robot.joint_damping[i]
        assert ankle.robot.joint_damping[i] == expected, name
        assert ankle.robot.prepare_state.damping[i] == expected, f"X uses the same gains: {name}"
    assert list(ankle.robot.joint_stiffness) == list(base.robot.joint_stiffness)
    assert list(ankle.robot.default_joint_pos) == list(base.robot.default_joint_pos)
    assert base.robot.joint_damping[names.index("left_ankle_roll_joint")] == 1.0
    registered = get_task("laai_sit_ankle_kd15")
    assert registered.robot.joint_damping[names.index("right_ankle_roll_joint")] == 1.5


def test_sit_task_without_a_policy_file_says_how_to_install_one(task, tmp_path):
    with pytest.raises(FileNotFoundError, match="robot/install.sh sit"):
        check_hold.make_controller(task, "puky_bike", "laai_sit", tmp_path / "missing.pt")
