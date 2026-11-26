"""Evaluate held-out joystick tracks, including forward and reverse performance."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from experiments.joystick.trajectory import SEGMENT_SECONDS

EVAL_SEED = 54821
EVAL_WORLDS = 32
MIN_SEGMENT_PROGRESS = 0.80
MAX_ENDPOINT_ERROR_M = 0.50
DIRECTION_DEADBAND_MPS = 0.05


def completed_segments(planned, progress, reached, error):
    return reached & (progress >= MIN_SEGMENT_PROGRESS * planned) & (error <= MAX_ENDPOINT_ERROR_M)


@torch.inference_mode()
def evaluate_policy(policy, device, *, worlds=EVAL_WORLDS, seed=EVAL_SEED):
    from experiments.joystick.task import COMMAND, joystick_env_cfg
    from experiments.push.task import bike_speed

    # The same clean scene and seed apply to both actors on the held-out tracks.
    cfg = joystick_env_cfg(play=True)
    cfg.scene.num_envs = worlds
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        obs, _ = env.reset(seed=seed)
        term = env.command_manager.get_term(COMMAND)
        reference = term.reference.clone()
        targets = term.feedforward[..., 0].clone()
        origin = term.origin.clone()
        world_heading = origin[:, None, 2] + reference[..., 2]
        heading_xy = torch.stack((world_heading.cos(), world_heading.sin()), dim=-1)
        c, s = origin[:, 2].cos(), origin[:, 2].sin()
        x, y = reference[:, -1, :2].unbind(dim=-1)
        endpoint = origin[:, :2] + torch.stack((x * c - y * s, x * s + y * c), dim=-1)
        active = torch.ones(worlds, dtype=torch.bool, device=device)
        survived = torch.zeros_like(active)
        path_sum = torch.zeros(worlds, device=device)
        path_count = torch.zeros_like(path_sum)
        endpoint_error = torch.zeros_like(path_sum)

        def zeros():
            return torch.zeros((worlds, len(SEGMENT_SECONDS)), device=device)

        progress, planned, segment_speed, segment_error = zeros(), zeros(), zeros(), zeros()
        reached = torch.zeros_like(progress, dtype=torch.bool)
        samples, speed_errors, direction_hits = (torch.zeros(2, device=device) for _ in range(3))
        segment_steps = [round(seconds / env.step_dt) for seconds in SEGMENT_SECONDS]
        for segment, values in enumerate(targets.split(segment_steps, dim=1)):
            planned[:, segment] = values.abs().sum(dim=1) * env.step_dt
            segment_speed[:, segment] = values[:, values.size(1) // 2]

        tick = 0
        for segment, steps in enumerate(segment_steps):
            for segment_step in range(steps):
                action = policy(TensorDict(obs, batch_size=[worlds]))
                obs, _, terminated, truncated, _ = env.step(action)
                ended = terminated | truncated
                survived |= active & truncated & ~terminated

                position = term.bike.data.root_link_pos_w[:, :2]
                endpoint_error = torch.where(
                    active, (position - endpoint).norm(dim=-1), endpoint_error
                )
                _, _, error = term.path_errors()
                path_sum += active * error
                path_count += active
                target = targets[:, tick]
                actual = bike_speed(env)
                velocity = term.bike.data.root_link_lin_vel_w[:, :2]
                progress[:, segment] += (
                    (velocity * heading_xy[:, tick]).sum(dim=-1)
                    * target.sign()
                    * active
                    * env.step_dt
                )
                choices = torch.stack(
                    (target > DIRECTION_DEADBAND_MPS, target < -DIRECTION_DEADBAND_MPS)
                )
                mask = choices & active
                samples += mask.sum(dim=1)
                speed_errors += (mask * (actual - target).abs()).sum(dim=1)
                direction_hits += (mask * (actual * target.sign() > DIRECTION_DEADBAND_MPS)).sum(
                    dim=1
                )

                tick += 1
                if segment_step + 1 == steps:
                    reached[:, segment] = active & ~terminated
                    segment_error[:, segment] = error
                active &= ~ended
                reset = ended.nonzero().flatten()
                if len(reset) and tick < term.steps:
                    obs, _ = env.reset(env_ids=reset)
                if not active.any():
                    break
            if not active.any():
                break

        completed = completed_segments(planned, progress, reached, segment_error)
        route_complete = (
            survived
            & ((planned <= 0) | completed).all(dim=1)
            & (endpoint_error <= MAX_ENDPOINT_ERROR_M)
        )

        def direction_report(direction, sign):
            expected = segment_speed * sign > DIRECTION_DEADBAND_MPS
            count = int(expected.sum())
            total = float(planned[expected].sum())
            samples_count = int(samples[direction])
            return {
                "segments": count,
                "segment_completion_fraction": float(completed[expected].sum()) / count
                if count
                else None,
                "signed_distance_ratio": float(progress[expected].sum()) / total if total else None,
                "speed_mae_mps": float(speed_errors[direction]) / samples_count
                if samples_count
                else None,
                "correct_direction_fraction": float(direction_hits[direction]) / samples_count
                if samples_count
                else None,
                "sample_count": samples_count,
            }

        return {
            "seed": seed,
            "worlds": worlds,
            "min_segment_progress": MIN_SEGMENT_PROGRESS,
            "max_endpoint_error_m": MAX_ENDPOINT_ERROR_M,
            "route": {
                "completion_fraction": float(route_complete.float().mean()),
                "survival_fraction": float(survived.float().mean()),
                "mean_endpoint_error_m": float(endpoint_error.mean()),
                "mean_path_error_m": float((path_sum / path_count.clamp_min(1)).mean()),
            },
            "forward": direction_report(0, 1),
            "reverse": direction_report(1, -1),
        }
    finally:
        env.close()


def main():
    from mjlab.rl import MjlabOnPolicyRunner

    from experiments.joystick.task import joystick_env_cfg, joystick_runner_cfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--worlds", type=int, default=EVAL_WORLDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.worlds < 1:
        parser.error("--worlds must be at least one")
    cfg = joystick_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner = MjlabOnPolicyRunner(
            RslRlVecEnvWrapper(env), asdict(joystick_runner_cfg(False)), device=args.device
        )
        runner.load(
            str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
        )
        policy = runner.get_inference_policy(device=args.device)
        report = evaluate_policy(policy, args.device, worlds=args.worlds)
    finally:
        env.close()
    result = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
