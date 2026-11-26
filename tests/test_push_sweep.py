"""The push reward sweep: reward sets, stages, command-line overrides, gates and ranking."""

import argparse
import json
import sys
from pathlib import Path

import pytest

from experiments.push import sweep

ROOT = Path(__file__).resolve().parents[1]


def test_reward_sets_parse_left_to_right_and_keep_the_core():
    assert sweep.parse_set("simple") == sweep.SETS["simple"]
    assert sweep.parse_set("simple+gait") == sweep.SETS["simple"] + sweep.GROUPS["gait"]
    terms = sweep.parse_set("all-form-arm_effort")
    assert "arm_effort" not in terms and not set(terms) & set(sweep.GROUPS["form"])
    assert sweep.parse_set("core+torque_near_limit") == (*sweep.CORE, "torque_near_limit")
    for bad in ("simple-failure", "simple+nonsense", "simple++gait", ""):
        with pytest.raises(ValueError):
            sweep.parse_set(bad)


def test_stages_have_the_expected_configs():
    groups = {c.name: c for c in sweep.make_stage("groups")}
    assert len(groups) == 2 + 2 * len(sweep.GROUPS)
    assert groups["all-no-form"].terms == sweep.parse_set("all-form")  # the Shaped task
    assert groups["simple"].terms == sweep.SETS["simple"]
    terms = sweep.make_stage("terms", "simple+balance")
    assert [c.name for c in terms] == ["base", "no-on_line", "no-saddle_load"] + [
        "no-action_rate_l2",
        "no-bike_upright",
    ]
    weights = {c.name: c for c in sweep.make_stage("weights", "core+bike_upright")}
    assert set(weights) == {
        "base",
        *(f"{t}-x{f:g}" for t in ("track_bike_speed", "bike_upright") for f in (0.5, 2.0)),
    }
    assert weights["bike_upright-x2"].scales["bike_upright"] == 2.0
    assert weights["bike_upright-x2"].terms == weights["base"].terms


def test_overrides_switch_off_and_rescale_only_what_changes():
    defaults = {t: 1.0 for t in sweep.ALL_TERMS} | {"failure": -250.0, "leg_band": -0.5}
    config = sweep.make_stage("weights", "simple+leg_band")
    by_name = {c.name: c for c in config}
    args = sweep.overrides(by_name["leg_band-x2"], defaults)
    pairs = dict(zip(args[::2], args[1::2], strict=True))
    assert pairs["--env.rewards.leg-band.weight"] == "-1"
    assert pairs["--env.rewards.step-symmetry.weight"] == "0"
    assert "--env.rewards.failure.weight" not in pairs  # unchanged
    assert all(v != "-0" for v in pairs.values())
    with pytest.raises(ValueError):
        sweep.overrides(by_name["base"], {"track_bike_speed": 2.0})


def test_overrides_match_the_form_task_rewards():
    pytest.importorskip("mjlab")
    from experiments.push import task as push

    rewards = push.push_env_cfg(shaped=True, form=True).rewards
    assert set(rewards) == set(sweep.ALL_TERMS)
    shaped = push.push_env_cfg(shaped=True).rewards
    assert set(sweep.parse_set("all-form")) == set(shaped)
    simple = push.push_env_cfg().rewards
    assert set(sweep.SETS["simple"]) == set(simple)


def _row(v, **over):
    row = {
        "command_mps": v,
        "success_fraction": 1.0,
        "mean_speed_mps": v,
        "speed_error_mps": 0.02 if v else 0.001,
        "lateral_deviation_m": 0.05,
        "torque_above_90_share": 0.0,
        "both_feet_off_share": 0.01,
        "bike_collision_share": 0.0,
        "step_phase_asymmetry": 0.05 if v else None,
        "step_force_asymmetry": 0.1 if v else None,
    }
    row.update(over)
    return row


def test_gates_fail_a_policy_on_the_torque_limit_or_hopping():
    good = sweep.judge({"policy": [_row(0.0), _row(0.2), _row(0.4)]})
    assert good["passed"] and good["quality"] == pytest.approx(0.925)
    # As the archived Simple run: tracks the speed, but at the torque limit and hopping.
    simple = sweep.judge(
        {
            "policy": [
                _row(0.0, torque_above_90_share=0.59),
                _row(0.2, torque_above_90_share=1.0, both_feet_off_share=0.39),
                _row(0.4, torque_above_90_share=1.0, both_feet_off_share=0.45),
            ]
        }
    )
    assert not simple["passed"]
    assert any("torque" in f for f in simple["failures"])
    assert any("hops" in f for f in simple["failures"])
    still = sweep.judge({"policy": [_row(0.0), _row(0.2, speed_error_mps=0.2)]})
    assert not still["passed"] and still["quality"] == pytest.approx(0.0)


def _write_run(root, run_name, rows):
    folder = root / f"2026-11-26_10-00-00_{run_name}"
    folder.mkdir()
    (folder / "evaluation.json").write_text(json.dumps({"policy": rows}))


def test_collect_ranks_passing_configs_by_fewest_terms(tmp_path):
    good = [_row(0.0), _row(0.2), _row(0.4)]
    bad = [_row(0.0), _row(0.2, both_feet_off_share=0.5), _row(0.4)]
    runs = []
    for config, terms, results in (
        ("big", sweep.SETS["all"], [good, good]),
        ("small", sweep.SETS["simple"], [good, good]),
        ("flaky", sweep.SETS["core"], [good, bad]),
        ("unfinished", sweep.SETS["core"], [good, None]),
    ):
        for seed, rows in enumerate(results):
            name = f"sw-{config}-s{seed}"
            runs.append({"config": config, "seed": seed, "run_name": name, "terms": list(terms)})
            if rows:
                _write_run(tmp_path, name, rows)
    ranked = sweep.collect({"runs": runs}, tmp_path)
    assert [r["config"] for r in ranked[:2]] == ["small", "big"]
    flaky = next(r for r in ranked if r["config"] == "flaky")
    assert flaky["passed"] == 1 and not flaky["passes_all_seeds"]
    unfinished = next(r for r in ranked if r["config"] == "unfinished")
    assert not unfinished["complete"] and unfinished["done"] == 1
    text = sweep.results_markdown(ranked, "sw")
    assert "Simplest passing set: " + " + ".join(sweep.SETS["simple"]) in text


def test_submit_builds_one_sbatch_job_per_run():
    pytest.importorskip("mjlab")
    sys.path.insert(0, str(ROOT / "scripts"))
    import sweep_push

    ns = argparse.Namespace(
        stage="terms",
        base="simple",
        only="base,no-on_line",
        seeds=2,
        seed=42,
        iterations=100,
        init_policy_from=None,
        local=False,
        cpu=False,
        num_envs=None,
        time="06:00:00",
        test_only=True,
    )
    name, runs = sweep_push.plan(ns)
    assert [(r["config"], r["seed"]) for r in runs] == [
        ("base", 42),
        ("base", 43),
        ("no-on_line", 42),
        ("no-on_line", 43),
    ]
    command = sweep_push.job_command(ns, name, runs[2])
    assert command[:2] == ["sbatch", "--parsable"] and "--test-only" in command
    assert "--time=06:00:00" in command
    assert f"--export=ALL,WANDB_RUN_GROUP={name},WANDB_TAGS=terms" in command
    i = command.index("cluster/train.slurm")
    assert command[i + 1] == "LAAI-K1-Push-Puky-Form"
    assert "--env.rewards.on-line.weight" in command
    assert command[command.index("--agent.run-name") + 1] == f"{name}-no-on_line-s42"
