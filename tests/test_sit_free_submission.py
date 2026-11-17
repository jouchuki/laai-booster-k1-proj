"""Paired submissions use identical training budgets and the student CPU launcher."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "submit_sit_free", ROOT / "scripts/submit_sit_free.py"
)
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def test_pair_has_same_budget_seed_task_and_separate_initialization():
    commands = submit.submission_commands(Path("checkpoint with spaces.pt"), "comparison")
    warm, scratch = commands.values()
    for command in (warm, scratch):
        assert command[0] == "sbatch"
        assert "cluster/train_cpu.slurm" in command
        assert "--time=12:00:00" in command
        assert command[command.index("--agent.seed") + 1] == "42"
        assert command[command.index("--agent.max-iterations") + 1] == "1500"
        assert command[command.index("--env.scene.num-envs") + 1] == "256"
        assert "LAAI-K1-Sit-Puky-Free" in command
        assert command[command.index("--agent.logger") + 1] == "wandb"
        assert not any(
            c.startswith(("--gres", "--gpus", "--qos", "--reservation")) for c in command
        )
    assert warm[-2:] == ["--agent.init-policy-from", "checkpoint with spaces.pt"]
    assert "--agent.init-policy-from" not in scratch
    assert "--init-from" not in warm + scratch, "resume would reuse old counters and optimizer"
    assert (
        warm[warm.index("--agent.run-name") + 1] != scratch[scratch.index("--agent.run-name") + 1]
    )


def test_test_only_does_not_submit_real_jobs():
    commands = submit.submission_commands(Path("checkpoint.pt"), "comparison", test_only=True)
    assert all("--test-only" in command for command in commands.values())
