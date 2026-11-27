"""Reward sweep for the push task: the simplest reward that still works.

Every run trains LAAI-K1-Push-Puky-Form (straight line) or LAAI-K1-Push-Puky-Steer (speed and
turn rate), whose rewards are the union of all candidate terms for that task. The two differ in
one task term: Form keeps the bike on its starting line (`on_line`), Steer tracks the commanded
turn rate (`track_bike_turn_rate`, a core term there). Sets and groups below name both; each
task uses the terms it has (TASK_TERMS).
A run switches terms off (weight 0, which mjlab's reward manager skips) or rescales them with
mjlab's command-line overrides (`--env.rewards.<term>.weight`), so every configuration is
the same code with different numbers, recorded in the run's params/env.yaml.

Stages, each started from the best result of the one before (scripts/sweep_push.py):

  1. groups:  which families of terms matter. All terms; Simple (task terms only); all minus
              one group; Simple plus one group.
  2. terms:   leave-one-out within a base set (the simplest passing set from stage 1).
  3. weights: each non-core term of a base set at 0.5x and 2x its default weight.

"Works" is a fixed set of pass/fail gates on the final evaluation (experiments/push/
evaluate.py): no falls, tracks the commanded speed (and turn rate), stays near the line when
going straight, never leans on the torque limit, does not hop, does not hit the bike. Among configurations that pass on every
seed, fewer active terms wins; the tracking quality breaks ties.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

FORM_TASK_ID = "LAAI-K1-Push-Puky-Form"  # as experiments/push/task.py (no mjlab import here)
STEER_TASK_ID = "LAAI-K1-Push-Puky-Steer"
TURN = "track_bike_turn_rate"  # the Steer task's turn-rate tracking
# The task itself: never switched off.
CORE = ("track_bike_speed", TURN, "failure")
TASK = ("on_line", "saddle_load", "action_rate_l2")  # the rest of the Simple task
GROUPS = {
    "gait": ("feet_air_time", "foot_slip", "both_feet_off", "feet_on_floor"),
    "posture": ("posture", "trunk_orientation"),
    "balance": ("bike_upright",),
    "motors": ("arm_effort", "action_acc_l2", "torque_near_limit"),
    "form": ("trunk_to_bike", "leg_band", "bike_collision", "step_symmetry"),
}
SETS = {
    "core": CORE,
    "task": TASK,
    "simple": CORE + TASK,
    "all": CORE + TASK + tuple(t for g in GROUPS.values() for t in g),
    **GROUPS,
}
ALL_TERMS = SETS["all"]
TASK_TERMS = {
    FORM_TASK_ID: tuple(t for t in ALL_TERMS if t != TURN),
    STEER_TASK_ID: tuple(t for t in ALL_TERMS if t != "on_line"),
}

# Pass/fail on the evaluation's policy rows. Moving rows are those with a non-zero command.
GATES = {
    "success_fraction_min": 0.95,  # every speed: episodes ending without a fall
    "speed_error_ratio_max": 0.3,  # moving: mean speed error at most 30% of the command
    "still_speed_max": 0.05,  # zero command: the bike stays put (m/s)
    "lateral_deviation_max": 0.3,  # m from the line (straight rows)
    "turn_rate_error_ratio_max": 0.3,  # turning: mean yaw rate within 30% of the command
    "torque_above_90_max": 0.02,  # share of time: the robot stops at 90% for 0.5 s
    "both_feet_off_max": 0.1,  # share of time: no hopping
    "bike_collision_max": 0.05,  # share of time touching the bike outside the saddle
}


@dataclass(frozen=True)
class Config:
    """One reward configuration: the active terms and their weight scales (1 = default)."""

    name: str
    scales: dict[str, float] = field(default_factory=dict)

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(t for t in ALL_TERMS if self.scales.get(t, 0.0) != 0.0)


def _terms_of(task: str) -> tuple[str, ...]:
    if task not in TASK_TERMS:
        raise ValueError(f"no sweep for task {task!r}; known: {', '.join(TASK_TERMS)}")
    return TASK_TERMS[task]


def parse_set(spec: str, task: str = FORM_TASK_ID) -> tuple[str, ...]:
    """Terms of a set expression such as "simple+gait+torque_near_limit" or "all-form-arm_effort".

    Each token is a set name (core, task, simple, all, or a group) or a term name; "+" adds it,
    "-" removes it, left to right. Core terms cannot be removed. Terms `task` does not have
    are left out (on_line for Steer, track_bike_turn_rate for Form).
    """
    available = _terms_of(task)
    tokens = re.findall(r"([+-]?)([a-z0-9_]+)", spec)
    if not tokens or "".join(s + n for s, n in tokens) != spec.replace(" ", ""):
        raise ValueError(f"cannot parse reward set {spec!r}")
    chosen: list[str] = []
    for sign, name in tokens:
        if name in SETS:
            names = SETS[name]
        elif name in available:
            names = (name,)
        else:
            known = ", ".join([*SETS, *available])
            raise ValueError(f"unknown reward set or term {name!r}; known: {known}")
        if sign == "-":
            if set(names) & set(CORE):
                raise ValueError(f"core terms {CORE} cannot be removed")
            chosen = [t for t in chosen if t not in names]
        else:
            chosen += [t for t in names if t not in chosen]
    return tuple(t for t in available if t in chosen)


def _config(name: str, terms, scales: dict[str, float] | None = None) -> Config:
    full = {t: 1.0 for t in terms}
    full.update(scales or {})
    return Config(name, full)


def stage_groups(task: str = FORM_TASK_ID) -> list[Config]:
    """All, Simple, all minus each group (all-no-form is the Shaped task), Simple plus each."""
    configs = [_config(name, parse_set(name, task)) for name in ("all", "simple")]
    for group in GROUPS:
        configs.append(_config(f"all-no-{group}", parse_set(f"all-{group}", task)))
    for group in GROUPS:
        configs.append(_config(f"simple-plus-{group}", parse_set(f"simple+{group}", task)))
    return configs


def stage_terms(base: str, task: str = FORM_TASK_ID) -> list[Config]:
    terms = parse_set(base, task)
    configs = [_config("base", terms)]
    for term in terms:
        if term not in CORE:
            configs.append(_config(f"no-{term}", [t for t in terms if t != term]))
    return configs


def stage_weights(base: str, task: str = FORM_TASK_ID, factors=(0.5, 2.0)) -> list[Config]:
    terms = parse_set(base, task)
    configs = [_config("base", terms)]
    for term in terms:
        if term == "failure":
            continue  # a fall's cost is a design choice, not a tuning knob
        for factor in factors:
            configs.append(_config(f"{term}-x{factor:g}", terms, {term: factor}))
    return configs


STAGES = {"groups": stage_groups, "terms": stage_terms, "weights": stage_weights}


def make_stage(stage: str, base: str = "all", task: str = FORM_TASK_ID) -> list[Config]:
    if stage == "groups":
        return stage_groups(task)
    return STAGES[stage](base, task)


def overrides(config: Config, defaults: dict[str, float]) -> list[str]:
    """mjlab command-line options that turn a sweep task's rewards (`defaults`: term weights)
    into `config`.

    Only terms whose weight changes are listed; switched-off terms get weight 0.
    """
    task_terms = next((t for t in TASK_TERMS.values() if set(t) == set(defaults)), None)
    if task_terms is None or not set(config.terms) <= set(task_terms):
        expected = " or ".join(str(sorted(set(t) ^ set(defaults))) for t in TASK_TERMS.values())
        raise ValueError(f"the task's rewards and the sweep's terms differ: {expected}")
    args: list[str] = []
    for term in task_terms:
        weight = defaults[term] * config.scales.get(term, 0.0) + 0.0  # + 0.0: no "-0"
        if weight != defaults[term]:
            args += [f"--env.rewards.{term.replace('_', '-')}.weight", f"{weight:g}"]
    return args


# --- Judging a finished run ----------------------------------------------------------------------
def _case(row: dict) -> str:
    turn = row.get("command_yaw_rate_rad_s", 0.0)
    return f"{row['command_mps']:g} m/s" + (f" {turn:+g} rad/s" if turn else "")


def judge(report: dict) -> dict:
    """Gates and a tracking-quality score from one run's evaluation.json.

    Quality is 1 - error / command, averaged over the moving rows; a turning row averages its
    speed and turn-rate scores. Rows without a turn rate (older reports) count as straight.
    """
    rows = report["policy"]
    failures = []
    moving = [r for r in rows if r["command_mps"] > 0]
    for r in rows:
        v, w, case = r["command_mps"], r.get("command_yaw_rate_rad_s", 0.0), _case(r)
        if r["success_fraction"] < GATES["success_fraction_min"]:
            failures.append(f"falls at {case} ({r['success_fraction']:.2f} survive)")
        lateral = r.get("lateral_deviation_m")
        if w == 0 and lateral is not None and lateral > GATES["lateral_deviation_max"]:
            failures.append(f"off the line at {case} ({lateral:.2f} m)")
        if r["torque_above_90_share"] > GATES["torque_above_90_max"]:
            failures.append(f"torque limit at {case} ({r['torque_above_90_share']:.0%})")
        if r["both_feet_off_share"] > GATES["both_feet_off_max"]:
            failures.append(f"hops at {case} ({r['both_feet_off_share']:.0%} feet off)")
        if r.get("bike_collision_share", 0.0) > GATES["bike_collision_max"]:
            failures.append(f"hits the bike at {case} ({r['bike_collision_share']:.0%})")
        if v == 0 and abs(r["mean_speed_mps"]) > GATES["still_speed_max"]:
            failures.append(f"moves when told to sit still ({r['mean_speed_mps']:.2f} m/s)")
        if v > 0 and r["speed_error_mps"] > GATES["speed_error_ratio_max"] * v:
            failures.append(f"speed at {case} off by {r['speed_error_mps']:.2f} m/s")
        if w and abs(r["mean_yaw_rate_rad_s"] - w) > GATES["turn_rate_error_ratio_max"] * abs(w):
            failures.append(f"turns at {r['mean_yaw_rate_rad_s']:+.2f} rad/s at {case}")

    def score(r):
        speed = 1.0 - min(1.0, r["speed_error_mps"] / r["command_mps"])
        w = r.get("command_yaw_rate_rad_s", 0.0)
        if not w:
            return speed
        return (speed + 1.0 - min(1.0, abs(r["mean_yaw_rate_rad_s"] - w) / abs(w))) / 2.0

    quality = sum(score(r) for r in moving) / len(moving) if moving else 0.0

    def worst(key, rows=rows, fn=max):
        values = [r[key] for r in rows if r.get(key) is not None]
        return fn(values) if values else None

    return {
        "passed": not failures,
        "failures": failures,
        "quality": quality,
        "worst_torque_above_90": worst("torque_above_90_share"),
        "worst_both_feet_off": worst("both_feet_off_share"),
        "worst_bike_collision": worst("bike_collision_share"),
        "worst_step_phase_asymmetry": worst("step_phase_asymmetry", moving),
        "worst_step_force_asymmetry": worst("step_force_asymmetry", moving),
        "worst_heading_drift_deg": worst("heading_drift_deg"),
    }


def find_run_dir(log_root: Path, run_name: str) -> Path | None:
    """The newest logs/laai_push/<date>_<run_name>/ folder, if any."""
    found = sorted(p for p in log_root.glob(f"*_{run_name}") if p.is_dir())
    return found[-1] if found else None


def collect(manifest: dict, log_root: Path) -> list[dict]:
    """One row per configuration, ranked: passing on every seed first, fewest terms first."""
    by_config: dict[str, dict] = {}
    for run in manifest["runs"]:
        row = by_config.setdefault(
            run["config"],
            {"config": run["config"], "terms": run["terms"], "seeds": [], "runs": []},
        )
        run_dir = find_run_dir(log_root, run["run_name"])
        evaluation = run_dir / "evaluation.json" if run_dir else None
        entry = {"run_name": run["run_name"], "seed": run["seed"], "status": "missing"}
        if evaluation and evaluation.is_file():
            entry.update(judge(json.loads(evaluation.read_text())), status="done")
            entry["run_dir"] = str(run_dir)
        elif run_dir:
            entry["status"] = "running or failed"
        row["runs"].append(entry)
        row["seeds"].append(run["seed"])
    rows = []
    for row in by_config.values():
        done = [r for r in row["runs"] if r["status"] == "done"]
        row["n_terms"] = len(row["terms"])
        row["done"] = len(done)
        row["passed"] = sum(r["passed"] for r in done)
        row["complete"] = row["done"] == len(row["runs"])
        row["passes_all_seeds"] = row["complete"] and row["passed"] == row["done"]
        row["quality"] = sum(r["quality"] for r in done) / len(done) if done else None
        row["failures"] = sorted({f for r in done for f in r["failures"]})
        rows.append(row)
    rows.sort(
        key=lambda r: (
            not r["passes_all_seeds"],
            r["n_terms"] if r["passes_all_seeds"] else 0,
            -(r["quality"] or 0.0),
        )
    )
    return rows


def results_markdown(rows: list[dict], sweep: str) -> str:
    lines = [
        f"# Push reward sweep {sweep}",
        "",
        (
            "Ranked: passing on every seed first, then fewest active terms, then speed "
            "(and turn-rate) tracking (1 = perfect)."
        ),
        "",
        "| Rank | Config | Terms | Passed | Quality | Failures |",
        "|---:|---|---:|---|---:|---|",
    ]
    for i, r in enumerate(rows, 1):
        quality = f"{r['quality']:.2f}" if r["quality"] is not None else "-"
        passed = f"{r['passed']}/{len(r['runs'])}" + ("" if r["complete"] else " (incomplete)")
        failures = "; ".join(r["failures"][:3]) + (" ..." if len(r["failures"]) > 3 else "")
        lines.append(
            f"| {i} | {r['config']} | {r['n_terms']} | {passed} | {quality} | {failures} |"
        )
    best = next((r for r in rows if r["passes_all_seeds"]), None)
    lines += ["", "Simplest passing set: " + (" + ".join(best["terms"]) if best else "none yet")]
    return "\n".join(lines) + "\n"
