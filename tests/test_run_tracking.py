"""Experiment tracking: W&B setup in scripts/train.py and scripts/run_summary.py."""

import importlib.util
from pathlib import Path

import pytest

# Training dependencies (`--group train`): CI installs only the dev group, so these skip there.
pytest.importorskip("mjlab")
pytest.importorskip("tensorboard")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_script("train")


@pytest.fixture(scope="module")
def summary():
    return load_script("run_summary")


def test_our_tasks_default_to_wandb(train):
    assert train.default_logger(["train.py", "LAAI-K1-Walk-Flat"]) == "wandb"
    assert train.default_logger(["train.py", "LAAI-K1-Sit-Puky"]) == "wandb"


def test_wandb_env_groups_by_experiment_and_tags_task(train):
    env = train.wandb_env(
        ["train.py", "LAAI-K1-Sit-Puky"], {}, logged_in=True, default_logger="wandb"
    )
    assert env["WANDB_RUN_GROUP"] == "laai_sit"
    assert env["WANDB_TAGS"] == "LAAI-K1-Sit-Puky"
    assert env["WANDB_DIR"].endswith("logs")
    assert "WANDB_MODE" not in env


def test_wandb_env_offline_without_key_and_respects_user_settings(train):
    argv = ["train.py", "LAAI-K1-Walk-Flat", "--agent.experiment-name=custom"]
    env = train.wandb_env(argv, {"WANDB_TAGS": "ablation"}, logged_in=False, default_logger="wandb")
    assert env["WANDB_MODE"] == "offline"
    assert env["WANDB_RUN_GROUP"] == "custom"
    assert env["WANDB_TAGS"] == "LAAI-K1-Walk-Flat,ablation"
    user_set = {"WANDB_MODE": "online", "WANDB_RUN_GROUP": "mine"}
    env = train.wandb_env(argv, user_set, logged_in=False, default_logger="wandb")
    assert "WANDB_MODE" not in env and "WANDB_RUN_GROUP" not in env


def test_wandb_env_empty_for_tensorboard_and_help(train):
    argv = ["train.py", "LAAI-K1-Walk-Flat", "--agent.logger", "tensorboard"]
    assert train.wandb_env(argv, {}, logged_in=True, default_logger="wandb") == {}
    argv = ["train.py", "LAAI-K1-Walk-Flat", "--help"]
    assert train.wandb_env(argv, {}, logged_in=True, default_logger="wandb") == {}


def test_run_summary_averages_last_iterations(summary, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tmp_path))
    for it in range(10):
        writer.add_scalar("Train/mean_reward", float(it), it)
        writer.add_scalar("Train/mean_episode_length", 100.0, it)
        writer.add_scalar("Episode_Reward/alive", 0.5, it)
        writer.add_scalar("Episode_Termination/fell_over", 0.0, it)
        writer.add_scalar("Metrics/twist/error_vel_xy", 0.2, it)
    writer.close()

    s = summary.summarize(summary.read_scalars(tmp_path), last=4)
    assert s["iterations"] == 10
    assert s["mean_reward"] == pytest.approx((6 + 7 + 8 + 9) / 4)
    assert s["peak_reward"] == 9
    assert s["rewards"] == {"alive": pytest.approx(0.5)}
    assert s["terminations"] == {"fell_over": 0.0}
    assert s["task_metrics"] == {"twist/error_vel_xy": pytest.approx(0.2)}
    block = summary.notes_block(s, tmp_path, "not logged to W&B")
    assert "| alive | 0.5 |" in block
    assert summary.table_row(s, tmp_path, "not logged to W&B").endswith("| - |")
