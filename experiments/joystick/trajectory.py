"""Sample joystick traces and integrate their commands with a planar Puky model."""

import math
from dataclasses import dataclass

import torch

WHEELBASE_M = 0.50
MAX_SPEED_MPS = 0.60
MAX_STEER_RAD = math.radians(35)
SEGMENT_SECONDS = (5.0, 10.0, 5.0)
EPISODE_SECONDS = sum(SEGMENT_SECONDS)
RAMP_SECONDS = 0.8
REVERSE_SHARE = 0.25  # fraction of moving segments sampled in reverse


@dataclass
class Trajectory:
    """Batched local reference poses (x, y, yaw), commands (m/s, rad) and stick (right, up)."""

    pose: torch.Tensor  # (worlds, steps + 1, 3), including the initial pose
    command: torch.Tensor  # (worlds, steps, 2), applied from pose[t] to pose[t + 1]
    stick: torch.Tensor  # (worlds, steps, 2), the raw direction request before ramping


def sample_trajectories(
    worlds: int,
    dt: float,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> Trajectory:
    """Latch each stick bearing relative to the heading at the start of its segment."""
    commands, sticks = [], []
    for seconds in SEGMENT_SECONDS:
        count = round(seconds / dt)
        ramp = (torch.arange(count, device=device) + 0.5) * dt
        ramp = (torch.minimum(ramp, count * dt - ramp) / RAMP_SECONDS).clamp(max=1.0)

        mode, effort, bearing, gear = torch.rand((4, worlds), device=device, generator=generator)
        direction = torch.where(gear < REVERSE_SHARE, -1.0, 1.0)
        cruise = torch.where(mode < 0.10, 0.0, 0.25 + 0.35 * effort)
        turn = (2.0 * bearing - 1.0) * math.radians(85)
        turn = torch.where(mode < 0.30, 0.0, turn)

        speed_magnitude = cruise[:, None] * ramp
        turn_distance = (speed_magnitude * ramp).sum(dim=1) * dt
        max_turn = turn_distance * math.tan(MAX_STEER_RAD) / WHEELBASE_M
        turn = torch.clamp(turn, -max_turn, max_turn)
        # In reverse the same wheel angle rotates the bike in the opposite direction.
        curvature = (direction * turn / turn_distance.clamp_min(1e-6))[:, None] * ramp
        steer = torch.atan(WHEELBASE_M * curvature)
        commands.append(torch.stack((direction[:, None] * speed_magnitude, steer), dim=-1))
        signed_radius = direction * cruise / MAX_SPEED_MPS
        stick = torch.stack((-signed_radius * turn.sin(), signed_radius * turn.cos()), dim=-1)
        sticks.append(stick[:, None, :].expand(-1, count, -1))

    command = torch.cat(commands, dim=1)
    distance = command[..., 0] * dt
    yaw_change = distance * command[..., 1].tan() / WHEELBASE_M
    yaw = yaw_change.cumsum(dim=1)
    midpoint = yaw - yaw_change / 2
    chord = distance * torch.sinc(yaw_change / (2 * math.pi))
    xy = torch.stack((chord * midpoint.cos(), chord * midpoint.sin()), dim=-1).cumsum(dim=1)
    pose = torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)
    return Trajectory(
        pose=torch.cat((torch.zeros_like(pose[:, :1]), pose), dim=1),
        command=command,
        stick=torch.cat(sticks, dim=1),
    )
