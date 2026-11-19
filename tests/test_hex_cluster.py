"""Hex student limits and batch-launch isolation; no training dependencies or GPU needed."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cluster/train.slurm"
CPU_SCRIPT = ROOT / "cluster/train_cpu.slurm"


def slurm_defaults(script=SCRIPT):
    return dict(
        line.removeprefix("#SBATCH --").split("=", 1)
        for line in script.read_text().splitlines()
        if line.startswith("#SBATCH --")
    )


def assert_student_request(options):
    assert options["nodes"] == options["ntasks"] == "1"
    if options["partition"] == "batch-gpu":
        assert options["gres"].startswith("gpu:")
        assert options["gres"].rsplit(":", 1)[-1] == "1"
        assert 1 <= int(options["cpus-per-task"]) <= 5
        max_days = 2
    else:
        assert options["partition"] in {"batch", "short", "long"}
        assert "gres" not in options and "gpus" not in options
        assert int(options["cpus-per-task"]) > 0
        max_days = {"batch": 4, "short": 2 / 24, "long": 28}[options["partition"]]
    days, _, clock = options["time"].rpartition("-")
    hours, minutes, seconds = map(int, clock.split(":"))
    duration = int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds
    assert 0 < duration <= max_days * 86400
    assert not {"exclusive", "oversubscribe", "qos", "reservation"} & options.keys()


@pytest.mark.parametrize("script", [SCRIPT, CPU_SCRIPT])
def test_batch_defaults_fit_hex_student_policy(script):
    assert_student_request(slurm_defaults(script))


@pytest.mark.parametrize("document", ["cluster/README.md", "docs/commands.md"])
def test_documented_submissions_and_allocations_fit_student_policy(document):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", (ROOT / document).read_text(), re.DOTALL):
        commands.extend(
            line
            for line in block.replace("\\\n", " ").splitlines()
            if line.startswith(("sbatch ", "srun "))
        )
    assert commands, f"No SLURM examples checked in {document}"
    for command in commands:
        script = CPU_SCRIPT if "cluster/train_cpu.slurm" in command else SCRIPT
        options = slurm_defaults(script)
        for token in shlex.split(command)[1:]:
            if token in {"cluster/train.slurm", "cluster/train_cpu.slurm"}:
                break  # Later arguments belong to training, not SLURM.
            if token.startswith("--"):
                name, _, value = token[2:].partition("=")
                options[name] = value
        assert_student_request(options)


@pytest.fixture
def batch(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("Batch launcher requires Bash")
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    (repo / "pyproject.toml").touch()
    (repo / "scripts/train.py").touch()
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "calls.jsonl"
    uv = binaries / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['MOCK_LOG'], 'a') as output:\n"
        "    output.write(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ), "
        "'cwd': os.getcwd()}) + '\\n')\n"
        "if '-c' in sys.argv and os.environ.get('MOCK_CUDA_FAIL'):\n"
        "    sys.exit(17)\n"
    )
    uv.chmod(0o755)
    smi = binaries / "nvidia-smi"
    smi.write_text("#!/bin/sh\nexit 0\n")
    smi.chmod(0o755)
    # Pass only shell essentials; never capture real account credentials in test logs.
    shell_env = {name: os.environ[name] for name in ("HOME", "PATH") if name in os.environ}
    env = dict(
        shell_env,
        PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
        SLURM_JOB_ID="123",
        SLURM_JOB_PARTITION="batch-gpu",
        SLURM_JOB_NUM_NODES="1",
        SLURM_NTASKS="1",
        SLURM_GPUS_ON_NODE="1",
        SLURM_CPUS_PER_TASK="3",
        SLURM_SUBMIT_DIR=str(repo),
        CUDA_VISIBLE_DEVICES="2",
        OPENBLAS_NUM_THREADS="128",
        MOCK_LOG=str(log),
    )
    return env, log


def run_batch(env, *arguments, script=SCRIPT):
    return subprocess.run(
        ["bash", str(script), "LAAI-K1-Sit-Puky-Lean", *arguments],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_batch_keeps_gpu_visibility_and_limits_threads(batch):
    env, log = batch
    arguments = ["--init-from", "logs/laai_sit/run with spaces/model_1499.pt"]
    result = run_batch(env, *arguments)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    training = calls[-1]
    assert training["args"][-len(arguments) :] == arguments
    assert "LAAI-K1-Sit-Puky-Lean" in training["args"]
    assert training["cwd"] == env["SLURM_SUBMIT_DIR"]
    assert training["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert "--no-sync" in training["args"]
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert training["env"][name] == "3"


@pytest.mark.parametrize("script", [SCRIPT, CPU_SCRIPT])
@pytest.mark.parametrize("entity", [None, "custom-team"])
def test_batch_preserves_team_default_and_allows_override(batch, script, entity):
    env, log = batch
    if script == CPU_SCRIPT:
        env.update(SLURM_JOB_PARTITION="batch", SLURM_GPUS_ON_NODE="0")
    if entity:
        env["WANDB_ENTITY"] = entity
    result = run_batch(env, script=script)
    assert result.returncode == 0, result.stderr
    training = json.loads(log.read_text().splitlines()[-1])
    assert training["env"]["WANDB_ENTITY"] == (entity or "K1Booster")


@pytest.mark.parametrize("script", [SCRIPT, CPU_SCRIPT])
def test_batch_loads_optional_dotenv_without_exposing_its_contents(batch, script):
    env, log = batch
    if script == CPU_SCRIPT:
        env.update(SLURM_JOB_PARTITION="batch", SLURM_GPUS_ON_NODE="0")
    # Dummy content only: never read real account credentials in tests.
    sentinel = "dummy-test-key"
    (Path(env["SLURM_SUBMIT_DIR"]) / ".env").write_text(f"WANDB_API_KEY={sentinel}\n")
    result = run_batch(env, script=script)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    for call in calls:
        index = call["args"].index("--env-file")
        assert call["args"][index + 1] == ".env"
    assert sentinel not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "changes",
    [
        {"SLURM_JOB_ID": ""},
        {"SLURM_JOB_PARTITION": "gpu"},
        {"SLURM_CPUS_PER_TASK": "6"},
        {"SLURM_CPUS_PER_TASK": "0"},
        {"SLURM_GPUS_ON_NODE": "0"},
        {"SLURM_GPUS_ON_NODE": "2"},
        {"SLURM_NTASKS": "2"},
        {"SLURM_JOB_NUM_NODES": "2"},
    ],
)
def test_batch_rejects_login_node_and_unsupported_allocations(batch, changes):
    env, log = batch
    result = run_batch(dict(env, **changes))
    assert result.returncode != 0
    assert not log.exists(), "Started Python before validating the allocation"


def test_cuda_failure_stops_before_training(batch):
    env, log = batch
    result = run_batch(dict(env, MOCK_CUDA_FAIL="1"))
    assert result.returncode == 17
    assert len(log.read_text().splitlines()) == 1


def test_cpu_batch_forces_cpu_and_forwards_training_options(batch):
    env, log = batch
    env.update(SLURM_JOB_PARTITION="batch", SLURM_CPUS_PER_TASK="4", SLURM_GPUS_ON_NODE="0")
    # Simulate a node with a visible physical GPU that this job is not allowed to use.
    result = run_batch(
        env, "--env.scene.num-envs", "8", "--agent.max-iterations", "3", script=CPU_SCRIPT
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1, "CPU job must not run a CUDA preflight"
    arguments = calls[0]["args"]
    assert arguments[arguments.index("--gpu-ids") + 1] == "None"
    assert arguments[-4:] == ["--env.scene.num-envs", "8", "--agent.max-iterations", "3"]
    assert calls[0]["env"]["OMP_NUM_THREADS"] == "4"
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "2"


@pytest.mark.parametrize(
    "changes, arguments",
    [
        ({"SLURM_JOB_ID": ""}, []),
        ({"SLURM_JOB_PARTITION": "batch-gpu"}, []),
        ({"SLURM_GPUS_ON_NODE": "1"}, []),
        ({"SLURM_JOB_GPUS": "0"}, []),
        ({"SLURM_NTASKS": "2"}, []),
        ({"SLURM_CPUS_PER_TASK": "0"}, []),
        ({}, ["--gpu-ids", "0"]),
        ({}, ["--gpu-ids=0"]),
    ],
)
def test_cpu_batch_rejects_invalid_allocations_and_gpu_overrides(batch, changes, arguments):
    env, log = batch
    env.update(SLURM_JOB_PARTITION="batch", SLURM_GPUS_ON_NODE="0")
    env.update(changes)
    result = run_batch(env, *arguments, script=CPU_SCRIPT)
    assert result.returncode != 0
    assert not log.exists()
