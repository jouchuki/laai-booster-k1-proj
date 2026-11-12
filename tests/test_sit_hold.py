"""The robot task "laai_sit_hold" (experiments/sit/deploy/laai_sit.py), rehearsed in the K1 +
bike scene. Needs booster_deploy (bash scripts/setup_booster.sh) and torch (uv sync --group sim);
skipped without them (CI clones only booster_assets)."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
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
