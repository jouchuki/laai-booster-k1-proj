"""Synchronized, split-screen CPU playback of the two free-bike policies.

Linux host, repository root (checkpoint paths are required):
    uv run --group train python scripts/compare_sit_free.py --pretrained PRETRAINED.pt --scratch SCRATCH.pt

Space pauses; Enter resets both sides; N changes the shared seed; Escape closes.
Each side stops at its first failure or at 10 seconds. The viewer uses the same seeded
one-world setup on both sides, deterministic actions, and identical pushes at 2/4/6/8 s.
These are visual trials, separate from the 32-world evaluation saved by the training job.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import glfw
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.sit import task as sit
from experiments.sit_free.runner import PolicyInitRunner, initialize_policy
from experiments.sit_free.task import bike_tilt_deg, free_sit_env_cfg, free_sit_runner_cfg
from experiments.sit_free.task import saddle_vertical_load_fraction as saddle_load


class Trial:
    def __init__(self, label, checkpoint, seed):
        self.label = label
        cfg = free_sit_env_cfg()
        cfg.scene.num_envs = 1
        cfg.seed = seed
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot")
        # Detect endings ourselves so a failed side stays visible instead of auto-resetting.
        cfg.terminations = {}
        self.env = ManagerBasedRlEnv(cfg, device="cpu")
        self.wrapped = RslRlVecEnvWrapper(self.env)
        runner = PolicyInitRunner(self.wrapped, asdict(free_sit_runner_cfg()), device="cpu")
        self.iteration = initialize_policy(runner, checkpoint)["source_iteration"]
        self.policy = runner.alg.get_policy().eval()
        self.obs, _ = self.wrapped.reset()
        self.steps = 0
        self.status = "Running"
        self.done = False
        self.model = self.env.sim.mj_model
        self.data = mujoco.MjData(self.model)

    @torch.inference_mode()
    def step(self, push=None):
        if self.done:
            return
        if push is not None:
            self.env.scene["robot"].write_root_link_velocity_to_sim(push)
        self.obs, _, _, _ = self.wrapped.step(self.policy(self.obs))
        self.steps += 1
        bike_fell = float(bike_tilt_deg(self.env)[0]) > sit.BIKE_FALL_DEG
        trunk_fell = float(sit.trunk_tilt_from_seated(self.env)[0]) > math.radians(35)
        if bike_fell or trunk_fell:
            self.status = "Bike fell" if bike_fell else "Torso limit"
            self.done = True
        elif self.steps >= 500:
            self.status, self.done = "Passed 10 seconds", True

    def sync_render_data(self):
        self.data.qpos[:] = self.env.sim.data.qpos[0].cpu().numpy()
        self.data.qvel[:] = self.env.sim.data.qvel[0].cpu().numpy()
        self.data.ctrl[:] = self.env.sim.data.ctrl[0].cpu().numpy()
        mujoco.mj_forward(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--once", action="store_true", help="close after one paired trial")
    parser.add_argument("--record", type=Path, help="save a 25 FPS side-by-side MP4")
    parser.add_argument("--screenshot", type=Path, help="save the last rendered frame")
    args = parser.parse_args()
    for checkpoint in (args.pretrained, args.scratch):
        if not checkpoint.is_file():
            parser.error(f"No checkpoint: {checkpoint}")
    if not glfw.init():
        raise RuntimeError("GLFW could not connect to the local display")
    window = glfw.create_window(1600, 800, "Free-bike comparison: pretrained | scratch", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create the comparison window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    state = {"paused": args.paused, "reset": True, "seed": args.seed, "mouse": None}
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -15
    trials, scenes, contexts = [], [], []
    writer = None
    last_pixels = None

    def on_key(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key in (glfw.KEY_ENTER, glfw.KEY_N):
            state["seed"] += int(key == glfw.KEY_N)
            state["reset"] = True
            state["paused"] = False

    def on_cursor(window, x, y):
        previous, state["mouse"] = state["mouse"], (x, y)
        if (
            previous is not None
            and scenes
            and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
        ):
            height = max(glfw.get_window_size(window)[1], 1)
            mujoco.mjv_moveCamera(
                trials[0].model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                (x - previous[0]) / height,
                (y - previous[1]) / height,
                scenes[0],
                camera,
            )

    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(
        window,
        lambda w, x, y: setattr(camera, "distance", max(0.5, camera.distance * math.exp(-0.1 * y))),
    )
    try:
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.record), fps=25, codec="libx264")
        while not glfw.window_should_close(window):
            started = time.monotonic()
            if state["reset"]:
                for context in contexts:
                    context.free()
                for trial in trials:
                    trial.env.close()
                trials = [
                    Trial(label, checkpoint, state["seed"])
                    for label, checkpoint in (
                        ("PRETRAINED", args.pretrained),
                        ("SCRATCH", args.scratch),
                    )
                ]
                scenes = [mujoco.MjvScene(t.model, maxgeom=10000) for t in trials]
                contexts = [
                    mujoco.MjrContext(t.model, mujoco.mjtFontScale.mjFONTSCALE_150) for t in trials
                ]
                pushes = (
                    2 * torch.rand(4, 1, 6, generator=torch.Generator().manual_seed(state["seed"]))
                    - 1
                )
                pushes *= torch.tensor([0.2, 0.2, 0.0, 0.3, 0.3, 0.3])
                step = 0
                state["reset"] = False
            advanced = False
            if not state["paused"] and not all(t.done for t in trials):
                push = pushes[step // 100 - 1] if step in (100, 200, 300, 400) else None
                for trial in trials:
                    trial.step(push)
                step += 1
                advanced = True
            width, height = glfw.get_framebuffer_size(window)
            for i, (trial, scene, context) in enumerate(zip(trials, scenes, contexts)):
                trial.sync_render_data()
                camera.lookat[:] = trial.env.scene["bike"].data.root_link_pos_w[0].cpu().numpy() + [
                    0.2,
                    0,
                    0.45,
                ]
                mujoco.mjv_updateScene(
                    trial.model,
                    trial.data,
                    mujoco.MjvOption(),
                    None,
                    camera,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    scene,
                )
                viewport = mujoco.MjrRect(i * (width // 2), 0, width // 2, height)
                mujoco.mjr_render(viewport, scene, context)
                text = (
                    f"{trial.label} | checkpoint {trial.iteration}\n"
                    f"{trial.status} | {trial.steps * 0.02:.2f} / 10.00 s | seed {state['seed']}\n"
                    f"Bike tilt {float(bike_tilt_deg(trial.env)[0]):.1f} deg | saddle support {float(saddle_load(trial.env)[0]):.0%}"
                )
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport,
                    text,
                    "",
                    context,
                )
                controls = "Space pause | Enter replay | N seed | Drag rotate | Scroll zoom"
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    viewport,
                    controls,
                    "",
                    context,
                )
            finished = all(t.done for t in trials)
            if (writer and advanced and step % 2 == 0) or (args.screenshot and finished):
                rgb = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), contexts[0])
                last_pixels = rgb[::-1].copy()
                if writer and advanced and step % 2 == 0:
                    writer.append_data(last_pixels)
            glfw.swap_buffers(window)
            glfw.poll_events()
            if finished and advanced:
                print(
                    json.dumps(
                        {t.label: {"seconds": t.steps * 0.02, "status": t.status} for t in trials}
                    ),
                    flush=True,
                )
                if args.once:
                    break
                state["paused"] = True
            time.sleep(max(0, 0.02 - (time.monotonic() - started)))
    finally:
        if writer:
            writer.close()
        if args.screenshot and last_pixels is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args.screenshot, last_pixels)
        for context in contexts:
            context.free()
        for trial in trials:
            trial.env.close()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
