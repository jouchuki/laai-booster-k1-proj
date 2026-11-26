"""Record a trained policy to an MP4 without a viewer window, at one fixed speed command.

One environment, no pushes or sensor noise, nominal saddle. The camera follows the robot.
The video is written next to the checkpoint unless --out is given.

Linux and macOS host, repository root (ordinary Python on both platforms):
    uv run --group train python scripts/record_video.py LAAI-K1-Push-Puky-Shaped \
        --checkpoint logs/laai_push/YOUR_RUN/model_2999.pt --seconds 20 --speed 0.3
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViewerConfig

import experiments  # noqa: F401  (registers every experiment's task with mjlab)

# Camera azimuth per view; the bike rides along the world +x axis.
VIEW_AZIMUTH = {"side": 90.0, "front": 180.0, "back": 0.0, "other_side": -90.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.3, help="commanded forward m/s")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--views",
        default="side",
        help=f"comma-separated camera angles placed side by side: {', '.join(VIEW_AZIMUTH)}",
    )
    args = parser.parse_args()
    views = [v.strip() for v in args.views.split(",")]
    if unknown := set(views) - set(VIEW_AZIMUTH):
        parser.error(f"unknown view(s) {sorted(unknown)}; choose from {list(VIEW_AZIMUTH)}")

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds + 1.0  # no reset inside the recording
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
    # A side view that follows the robot, over a checkered floor so the motion is visible.
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
    cfg.viewer.entity_name, cfg.viewer.body_name = "robot", "trunk"
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.5, -15.0, 90.0
    scene_fn = cfg.scene.spec_fn

    def checkered_floor(spec: mujoco.MjSpec) -> None:
        if scene_fn is not None:
            scene_fn(spec)
        spec.add_texture(
            name="video_grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.45, 0.5, 0.55],
            rgb2=[0.6, 0.65, 0.7],
            width=512,
            height=512,
        )
        material = spec.add_material(name="video_grid", texrepeat=[2, 2], texuniform=True)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "video_grid"
        for geom in spec.geoms:
            if geom.name == "floor":
                geom.material = "video_grid"
                geom.rgba = [1, 1, 1, 1]

    cfg.scene.spec_fn = checkered_floor
    cfg.observations["actor"].enable_corruption = False
    for name in ("push_robot", "saddle_position"):
        cfg.events.pop(name, None)
    cfg.curriculum = {}
    for command in cfg.commands.values():  # a fixed forward speed, if the task has one
        if hasattr(command, "ranges") and hasattr(command.ranges, "lin_vel_x"):
            command.ranges.lin_vel_x = (args.speed, args.speed)
            command.rel_standing_envs = 0.0
            command.resampling_time_range = (1e6, 1e6)

    env = ManagerBasedRlEnv(cfg, device="cpu", render_mode="rgb_array")
    agent_cfg = load_rl_cfg(args.task)
    if hasattr(agent_cfg, "init_policy_from"):
        agent_cfg.init_policy_from = None  # the checkpoint below replaces the actor anyway
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    camera = env._offline_renderer._cam  # the tracking camera env.render() uses

    def render_views() -> np.ndarray:
        images = []
        for view in views:
            camera.azimuth = VIEW_AZIMUTH[view]
            images.append(env.render())
        return np.concatenate(images, axis=1)

    obs = wrapped.get_observations()
    frames = []
    with torch.inference_mode():
        for _ in range(round(args.seconds / env.step_dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            frames.append(render_views())
            if dones.any():
                print("[WARN] The episode ended early (a fall?); the video stops here.")
                break
    env.close()

    out = args.out or args.checkpoint.parent / "videos" / f"{args.speed:g}mps_{args.seconds:g}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=round(1.0 / env.step_dt))
    print(f"[INFO] Wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
