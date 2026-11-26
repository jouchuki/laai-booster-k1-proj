"""Build a standalone HTML preview from the training sampler."""

import argparse
import json
from pathlib import Path

import torch

from experiments.joystick.trajectory import (
    EPISODE_SECONDS,
    MAX_SPEED_MPS,
    SEGMENT_SECONDS,
    sample_trajectories,
)

DEFAULT_SEED = 5
EXAMPLES_PER_MANEUVER = {
    "Forward straight": 2,
    "Forward left arc": 3,
    "Forward right arc": 3,
    "Forward S-curve": 3,
    "Stop and restart": 3,
    "Straight reverse": 1,
    "Reverse curve": 2,
    "Forward and reverse": 1,
}


def maneuver(segments: list[list[float]]) -> str:
    speed, steer = zip(*segments, strict=True)
    if min(speed) < 0:
        if speed[1] < 0:
            return "Straight reverse" if abs(steer[1]) < 1e-6 else "Reverse curve"
        return "Forward and reverse"
    if speed[0] > 0 and speed[1] == 0 and speed[2] > 0:
        return "Stop and restart"
    if min(steer) < -1e-6 and max(steer) > 1e-6:
        return "Forward S-curve"
    if max(steer) > 1e-6:
        return "Forward left arc"
    if min(steer) < -1e-6:
        return "Forward right arc"
    return "Forward straight" if max(speed) > 0 else "Stopped"


def preview_data(seed: int) -> dict:
    dt, stride = 0.02, 5
    sampled = sample_trajectories(256, dt, generator=torch.Generator().manual_seed(seed))
    midpoints = []
    elapsed = 0.0
    for duration in SEGMENT_SECONDS:
        midpoints.append(round((elapsed + duration / 2) / dt))
        elapsed += duration

    groups = {label: [] for label in EXAMPLES_PER_MANEUVER}
    for index, segments in enumerate(sampled.command[:, midpoints].tolist()):
        label = maneuver(segments)
        if label in groups:
            groups[label].append(index)

    selected, spare = [], []
    for label, limit in EXAMPLES_PER_MANEUVER.items():
        group = [(label, index) for index in groups[label]]
        selected.extend(group[:limit])
        spare.extend(group[limit:])
    missing = sum(EXAMPLES_PER_MANEUVER.values()) - len(selected)
    selected.extend(spare[:missing])
    order = {label: rank for rank, label in enumerate(EXAMPLES_PER_MANEUVER)}
    selected.sort(key=lambda example: order[example[0]])
    index = [world for _, world in selected]
    values = torch.cat((sampled.command[index], sampled.stick[index]), dim=-1)
    # The terminal pose has no subsequent command.
    values = torch.cat((values, torch.zeros_like(values[:, :1])), dim=1)
    rows = torch.cat((sampled.pose[index], values), dim=-1)[:, ::stride]
    episodes = [
        {"label": f"{label} · {number:02d}", "rows": [[round(x, 6) for x in row] for row in path]}
        for number, ((label, _), path) in enumerate(zip(selected, rows.tolist(), strict=True), 1)
    ]
    return {
        "dt": dt * stride,
        "duration": EPISODE_SECONDS,
        "segments": SEGMENT_SECONDS,
        "maxSpeed": MAX_SPEED_MPS,
        "episodes": episodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    template = Path(__file__).with_name("preview.html").read_text()
    args.output.write_text(
        template.replace("__TRAJECTORIES__", json.dumps(preview_data(args.seed)))
    )
