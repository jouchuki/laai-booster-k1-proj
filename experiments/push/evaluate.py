"""The same test for every push task: fixed speed commands, deterministic actions.

Every world gets one 10 s episode per commanded speed, without pushes or sensor noise, for the
policy and for a zero-action baseline (the robot holding the seated pose, which should not
move the bike). Measurements start after `WARMUP_S`, so the start from rest is not counted.
The numbers are physical quantities, independent of the reward, so the simple and the shaped
policy can be compared row by row.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from tensordict import TensorDict

EVAL_SPEEDS_MPS = (0.0, 0.2, 0.4)
EVAL_SECONDS = 10.0
WARMUP_S = 2.0


def _make_env(device, num_envs, seed):
    # Import here to avoid a cycle during task registration.
    from experiments.push.task import push_env_cfg

    cfg = push_env_cfg()  # the scene, sensors and terminations are the same for both tasks
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = EVAL_SECONDS
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot")
    cfg.events.pop("saddle_position")  # the nominal saddle, so every run is the same test
    cfg.curriculum = {}
    return ManagerBasedRlEnv(cfg, device=str(device))


@torch.inference_mode()
def _run(env, policy, device, num_envs, seed, speed):
    """One episode per world at one commanded speed; `policy` None means zero actions."""
    from experiments.push import task as push

    command = env.command_manager.get_term(push.COMMAND).cfg
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (speed, speed)
    obs, _ = env.reset(seed=seed)
    zero = torch.zeros(num_envs, env.action_manager.total_action_dim, device=device)
    complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success = torch.zeros_like(complete)
    was_down = push._feet_down(env)
    measured = torch.zeros(num_envs, device=device)  # steps counted after the warm-up
    totals = {
        name: torch.zeros(num_envs, device=device)
        for name in (
            "speed",
            "speed_error",
            "saddle_load",
            "both_feet_off",
            "foot_slip",
            "action_change",
            "torque_above_90",
            "touchdowns",
            "bike_collision",
            "trunk_bike_yaw",
            "knee_out",
            "foot_out",
        )
    }
    # Step asymmetry only means something while both feet have a current step.
    step_totals = {k: torch.zeros(num_envs, device=device) for k in ("air", "force", "phase")}
    step_count = torch.zeros(num_envs, device=device)
    distance = torch.zeros(num_envs, device=device)
    lateral = torch.zeros(num_envs, device=device)
    heading = torch.zeros(num_envs, device=device)
    bike_falls = torch.zeros_like(complete)
    torque_max = torch.zeros(num_envs, device=device)
    warmup_steps = round(WARMUP_S / env.step_dt)
    for step in range(env.max_episode_length):
        actions = zero if policy is None else policy(TensorDict(obs, batch_size=[num_envs]))
        obs, _, terminated, truncated, _ = env.step(actions)
        finished = ~complete & (terminated | truncated)
        success |= finished & truncated & ~terminated
        bike_falls |= finished & env.termination_manager.get_term("bike_fell")
        complete |= finished
        # A world that just finished has been reset: its state belongs to a new episode.
        live = ~complete
        down = push._feet_down(env)
        if step >= warmup_steps:
            on = live.float()
            measured += on
            totals["speed"] += on * push.bike_speed(env)
            totals["speed_error"] += on * push.speed_error(env)
            totals["saddle_load"] += on * push.saddle_vertical_load_fraction(env)
            totals["both_feet_off"] += on * push.both_feet_off(env)
            totals["foot_slip"] += on * push.foot_slip_speed(env)
            totals["action_change"] += on * push.sit.action_change(env)
            totals["torque_above_90"] += on * push.torque_above_90(env)
            totals["touchdowns"] += on * (down & ~was_down).sum(-1)
            totals["bike_collision"] += on * push.bike_collision_any(env)
            totals["trunk_bike_yaw"] += on * push.trunk_bike_yaw_deg(env)
            totals["knee_out"] += on * push.knee_out_cm(env)
            totals["foot_out"] += on * push.foot_out_cm(env)
            asym = push.StepTracker.of(env).asymmetry()
            counted = on * asym["valid"].float()
            step_count += counted
            for key, total in step_totals.items():
                total += counted * asym[key]
            # After the warm-up: settling into the saddle at the start briefly hits a limit.
            peak = torch.maximum(torque_max, push.torque_ratio_max(env))
            torque_max = torch.where(live, peak, torque_max)
        distance = torch.where(live, push.bike_distance(env), distance)
        lateral = torch.where(live, push.lateral_deviation(env), lateral)
        heading = torch.where(live, push.heading_error_deg(env), heading)
        was_down = down
        if complete.all():
            break
    steps = measured.clamp(min=1.0)
    mean = {name: float((total / steps).mean()) for name, total in totals.items()}
    stepped = step_count > 0
    step_mean = {
        key: float((total[stepped] / step_count[stepped]).mean()) if stepped.any() else None
        for key, total in step_totals.items()
    }
    return {
        "command_mps": speed,
        "success_fraction": float(success.float().mean()),
        "mean_speed_mps": mean["speed"],
        "speed_error_mps": mean["speed_error"],
        "bike_fall_fraction": float(bike_falls.float().mean()),
        "distance_m": float(distance.mean()),
        "lateral_deviation_m": float(lateral.mean()),
        "heading_error_deg": float(heading.mean()),
        "saddle_load": mean["saddle_load"],
        "both_feet_off_share": mean["both_feet_off"],
        "foot_slip_mps": mean["foot_slip"],
        "steps_per_second": mean["touchdowns"] / env.step_dt,
        "action_change": mean["action_change"],
        "torque_ratio_max": float(torque_max.max()),
        "torque_above_90_share": mean["torque_above_90"],
        "bike_collision_share": mean["bike_collision"],
        "trunk_bike_yaw_deg": mean["trunk_bike_yaw"],
        "knee_out_cm": mean["knee_out"],
        "foot_out_cm": mean["foot_out"],
        # None when no world stepped with both feet (e.g. sitting still, zero actions).
        "step_air_asymmetry": step_mean["air"],
        "step_force_asymmetry": step_mean["force"],
        "step_phase_asymmetry": step_mean["phase"],
    }


def evaluate_policy(policy, device, *, speeds=EVAL_SPEEDS_MPS, num_envs=32, seed=12345):
    """Results per commanded speed, for `policy` and for zero actions."""
    env = _make_env(device, num_envs, seed)
    try:
        return {
            "seed": seed,
            "episodes_per_speed": num_envs,
            "episode_seconds": EVAL_SECONDS,
            "policy": [_run(env, policy, device, num_envs, seed, v) for v in speeds],
            "zero_action": [_run(env, None, device, num_envs, seed, v) for v in speeds],
        }
    finally:
        env.close()
