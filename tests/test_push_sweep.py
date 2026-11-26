"""The push reward sweep: reward sets, stages, command-line overrides, gates and ranking."""

import argparse
import json
import sys
from pathlib import Path

import pytest

from experiments.push import sweep

ROOT = Path(__file__).resolve().parents[1]


def test_reward_sets_parse_left_to_right_and_keep_the_core():
    simple = ("track_bike_speed", "failure", "on_line", "saddle_load", "action_rate_l2")
    assert sweep.parse_set("simple") == simple
    assert sweep.parse_set("simple+gait") == simple + sweep.GROUPS["gait"]
    terms = sweep.parse_set("all-form-arm_effort")
    assert "arm_effort" not in terms and not set(terms) & set(sweep.GROUPS["form"])
    assert sweep.parse_set("core+torque_near_limit") == (
        "track_bike_speed",
        "failure",
        "torque_near_limit",
    )
    for bad in ("simple-failure", "simple+nonsense", "simple++gait", ""):
        with pytest.raises(ValueError):
            sweep.parse_set(bad)


def test_steer_sets_swap_the_line_for_the_turn_rate():
    steer = sweep.STEER_TASK_ID
    assert sweep.parse_set("core", steer) == ("track_bike_speed", sweep.TURN, "failure")
    simple = sweep.parse_set("simple", steer)
    assert "on_line" not in simple and sweep.TURN in simple
    assert sweep.TURN not in sweep.parse_set("all")  # the Form task has no turn rate
    with pytest.raises(ValueError):
        sweep.parse_set("simple-track_bike_turn_rate", steer)  # core
    with pytest.raises(ValueError):
        sweep.parse_set("simple+on_line", steer)
    terms = [c.name for c in sweep.make_stage("terms", "simple", steer)]
    assert terms == ["base", "no-saddle_load", "no-action_rate_l2"]
    groups = {c.name: c for c in sweep.make_stage("groups", task=steer)}
    assert groups["all"].terms == sweep.TASK_TERMS[steer]
    defaults = {t: 1.0 for t in sweep.TASK_TERMS[steer]}
    args = sweep.overrides(groups["simple"], defaults)
    assert "--env.rewards.on-line.weight" not in args
    assert "--env.rewards.track-bike-turn-rate.weight" not in args  # kept at its default
    with pytest.raises(ValueError):  # a Form config against the Steer task's rewards
        sweep.overrides(sweep.make_stage("groups")[1], defaults)


def test_stages_have_the_expected_configs():
    groups = {c.name: c for c in sweep.make_stage("groups")}
    assert len(groups) == 2 + 2 * len(sweep.GROUPS)
    assert groups["all-no-form"].terms == sweep.parse_set("all-form")  # the Shaped task
    assert groups["simple"].terms == sweep.parse_set("simple")
    assert "track_bike_turn_rate" not in groups["all"].terms
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
    form_terms = sweep.TASK_TERMS[sweep.FORM_TASK_ID]
    defaults = {t: 1.0 for t in form_terms} | {"failure": -250.0, "leg_band": -0.5}
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
    assert set(rewards) == set(sweep.TASK_TERMS[push.FORM_TASK_ID])
    shaped = push.push_env_cfg(shaped=True).rewards
    assert set(sweep.parse_set("all-form")) == set(shaped)
    simple = push.push_env_cfg().rewards
    assert set(sweep.parse_set("simple")) == set(simple)
    assert set(push.push_env_cfg(steer=True).rewards) == set(sweep.TASK_TERMS[push.STEER_TASK_ID])
    assert (sweep.FORM_TASK_ID, sweep.STEER_TASK_ID) == (push.FORM_TASK_ID, push.STEER_TASK_ID)


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


def _turn(v, w, yaw, **over):
    return _row(
        v,
        command_yaw_rate_rad_s=w,
        mean_yaw_rate_rad_s=yaw,
        lateral_deviation_m=None,
        heading_drift_deg=10.0,
        **over,
    )


def test_gates_check_the_turn_rate_and_skip_the_line_when_turning():
    rows = [_row(0.0), _row(0.2), _turn(0.2, 0.1, 0.09), _turn(0.4, -0.2, -0.2)]
    good = sweep.judge({"policy": rows})
    assert good["passed"], good["failures"]
    assert good["quality"] == pytest.approx((0.9 + (0.9 + 0.9) / 2 + (0.95 + 1.0) / 2) / 3)
    assert good["worst_heading_drift_deg"] == 10.0
    straight = sweep.judge({"policy": [*rows[:2], _turn(0.2, 0.1, 0.0)]})
    assert not straight["passed"]
    assert straight["failures"] == ["turns at +0.00 rad/s at 0.2 m/s +0.1 rad/s"]
    wrong_way = sweep.judge({"policy": [_turn(0.4, -0.2, 0.2)]})
    assert not wrong_way["passed"]
    drifting = sweep.judge({"policy": [_row(0.2, lateral_deviation_m=0.5)]})
    assert any("off the line" in f for f in drifting["failures"])


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
        task=sweep.FORM_TASK_ID,
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

    ns.task, ns.only = sweep.STEER_TASK_ID, "base,no-saddle_load"
    name, runs = sweep_push.plan(ns)
    assert name.startswith("sweep-steer-terms-")
    command = sweep_push.job_command(ns, name, runs[2])
    assert command[command.index("cluster/train.slurm") + 1] == "LAAI-K1-Push-Puky-Steer"
    assert "--env.rewards.saddle-load.weight" in command
    assert "--env.rewards.on-line.weight" not in command
