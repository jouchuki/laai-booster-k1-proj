"""Linux desktop playback with sixteen MuJoCo viewports and per-world path overlays."""

import argparse
import ctypes as ct
import math
import sys
import time
from collections import deque
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import glfw
import mujoco as mj
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.joystick.task import COMMAND, SHAPED_TASK_ID, TASK_ID
from experiments.joystick.trajectory import MAX_SPEED_MPS, SEGMENT_SECONDS

N = 16
TITLE = "Puky Joystick | 16 MuJoCo simulations"
ACTUAL = (1.0, 0.74, 0.30)
REFERENCE = (0.35, 0.88, 1.0)
MUTED = (0.54, 0.63, 0.70)
PANEL = (0.05, 0.09, 0.13)
GL = ct.CDLL("libGL.so.1")
LINE_LOOP, LINE_STRIP, TRIANGLE_FAN, QUADS = 0x0002, 0x0003, 0x0006, 0x0007
PROJECTION, MODELVIEW = 0x1701, 0x1700
OVERLAY_DISABLED_FLAGS = (
    0x0B71,  # depth test
    0x0B50,  # lighting
    0x0DE1,  # texture 2D
    0x0C11,  # scissor test
    0x0BE2,  # blending
    0x0B44,  # face culling
    0x0BC0,  # alpha test
)


def bind(name, *args):
    function = getattr(GL, name)
    function.argtypes = args
    function.restype = None
    return function


begin = bind("glBegin", ct.c_uint)
end = bind("glEnd")
vertex = bind("glVertex2f", ct.c_float, ct.c_float)
color = bind("glColor3f", ct.c_float, ct.c_float, ct.c_float)
linewidth = bind("glLineWidth", ct.c_float)
viewport = bind("glViewport", *([ct.c_int] * 4))
matrix = bind("glMatrixMode", ct.c_uint)
identity = bind("glLoadIdentity")
pushmatrix = bind("glPushMatrix")
popmatrix = bind("glPopMatrix")
ortho = bind("glOrtho", *([ct.c_double] * 6))
disable = bind("glDisable", ct.c_uint)
pushattrib = bind("glPushAttrib", ct.c_uint)
popattrib = bind("glPopAttrib")


def line(points, rgb, width=1, mode=LINE_STRIP):
    if len(points) < 2:
        return
    color(*rgb)
    linewidth(width)
    begin(mode)
    for x, y in points:
        vertex(x, y)
    end()


def disk(x, y, r, rgb, mode=TRIANGLE_FAN):
    color(*rgb)
    begin(mode)
    if mode == TRIANGLE_FAN:
        vertex(x, y)
    for angle in np.linspace(0, 2 * math.pi, 25):
        vertex(x + r * math.cos(angle), y + r * math.sin(angle))
    end()


def panel(x, y, width, height):
    line([(x, y), (x + width, y), (x + width, y + height), (x, y + height)], PANEL, mode=QUADS)


def overlay(ctx, box, label):
    mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_TOPLEFT, box, label, "", ctx)


def turn_direction(reference, feedforward, step, cuts):
    segment = min(int(np.searchsorted(cuts[1:], step, side="right")), len(cuts) - 2)
    start, end = cuts[segment : segment + 2]
    cruise = feedforward[(start + end) // 2, 0]
    turn = reference[end, 2] - reference[start, 2]
    radius = cruise / MAX_SPEED_MPS
    return np.array([-radius * math.sin(turn), radius * math.cos(turn)])


def topdown(reference, trace, actual, heading, stick, tile):
    w, h = tile.width, tile.height
    pushattrib(0x000FFFFF)
    viewport(tile.left, tile.bottom, w, h)
    for flag in OVERLAY_DISABLED_FLAGS:
        disable(flag)
    matrix(PROJECTION)
    pushmatrix()
    identity()
    ortho(0, w, 0, h, -1, 1)
    matrix(MODELVIEW)
    pushmatrix()
    identity()

    map_w, map_h = min(w * 0.33, 122), min(h * 0.49, 114)
    ox, oy = w - map_w - 8, 8
    panel(ox, oy, map_w, map_h)
    inset = 10
    points = np.concatenate((reference[:, :2], np.asarray(trace), actual[None]))
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = min(
        (map_w - 2 * inset) / max(0.35, hi[1] - lo[1]),
        (map_h - 2 * inset) / max(0.35, hi[0] - lo[0]),
    )
    center = (lo + hi) / 2

    def screen(xy):
        local = np.asarray(xy) - center
        return np.column_stack(
            (ox + map_w / 2 - local[:, 1] * scale, oy + map_h / 2 + local[:, 0] * scale)
        )

    planned = screen(reference[:, :2])
    line(planned, REFERENCE, 1.6)
    line(screen(trace), ACTUAL, 2.1)
    px, py = screen(actual[None])[0]
    disk(px, py, 3.0, ACTUAL)
    line([(px, py), (px - 7 * math.sin(heading), py + 7 * math.cos(heading))], ACTUAL, 2)

    size = min(56, h * 0.32)
    sx, sy = w - size - 8, h - size - 8
    panel(sx, sy, size, size)
    x, y, radius = sx + size / 2, sy + size / 2, size / 2 - 7
    disk(x, y, radius, MUTED, mode=LINE_LOOP)
    line([(x - radius, y), (x + radius, y)], MUTED)
    line([(x, y - radius), (x, y + radius)], MUTED)
    marker = (x + radius * stick[0], y + radius * stick[1])
    line([(x, y), marker], REFERENCE, 2)
    disk(*marker, 3.4, REFERENCE)

    popmatrix()
    matrix(PROJECTION)
    popmatrix()
    matrix(MODELVIEW)
    popattrib()


def draw_reference(scene, points):
    for start, finish in pairwise(points):
        if np.linalg.norm(finish - start) < 0.006:
            continue
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_CYLINDER,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.24, 0.76, 0.88, 0.82], dtype=np.float32),
        )
        geom.category = mj.mjtCatBit.mjCAT_DECOR
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CYLINDER, 0.009, start, finish)
        scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", choices=(TASK_ID, SHAPED_TASK_ID), default=TASK_ID)
    parser.add_argument("--seed", type=int, default=int(time.time()) % (2**31 - 1))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = N
    cfg.seed = args.seed
    print(f"Sampling sixteen independent tracks, seed {cfg.seed}", flush=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    agent_cfg = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device="cpu")
    checkpoint = args.checkpoint
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    term = env.command_manager.get_term(COMMAND)

    if not glfw.init():
        raise RuntimeError("GLFW could not connect to the desktop")
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    window = glfw.create_window(1730, 950, TITLE, None, None)
    if window is None:
        raise RuntimeError("GLFW could not create the demo window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    glfw.set_window_pos(window, 85, 170)
    requested = {"paused": False, "reset": False}

    def on_key(_window, key, _scan, action, _mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            requested["paused"] = not requested["paused"]
        elif key == glfw.KEY_R:
            requested["reset"] = True

    glfw.set_key_callback(window, on_key)
    model = env.sim.mj_model
    data = mj.MjData(model)
    scene = mj.MjvScene(model, maxgeom=1200)
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100)
    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, ctx)
    option = mj.MjvOption()
    option.geomgroup[3:] = 0
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 2.95, -34, 135
    scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
    scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
    cuts = np.cumsum([0, *[round(s / env.step_dt) for s in SEGMENT_SECONDS]])
    traces = [deque(maxlen=600) for _ in range(N)]
    frames = 0
    next_frame = time.monotonic()

    try:
        with torch.inference_mode():
            obs, _ = wrapped.reset()
            print(f"Loaded {checkpoint.name}. 16 MuJoCo worlds and 16 policy inputs.", flush=True)
            while not glfw.window_should_close(window):
                glfw.poll_events()
                if requested["reset"]:
                    obs, _ = wrapped.reset()
                    for trail in traces:
                        trail.clear()
                    requested["reset"] = False

                if not requested["paused"]:
                    for _ in range(2):
                        obs, _, dones, _ = wrapped.step(policy(obs))
                        for idx in dones.nonzero().flatten().tolist():
                            traces[idx].clear()

                width, height = glfw.get_framebuffer_size(window)
                if min(width, height) < 300:
                    glfw.wait_events_timeout(0.15)
                    continue
                pose = env.sim.data.qpos.cpu().numpy()
                positions = term.bike.data.root_link_pos_w.cpu().numpy()
                headings = term.bike.data.heading_w.cpu().numpy()
                speed = term.bike.data.root_link_lin_vel_b[:, 0].cpu().numpy()
                reference = term.reference.cpu().numpy()
                feedforward = term.feedforward.cpu().numpy()
                origin = term.origin.cpu().numpy()
                commands = term.command.cpu().numpy()
                index = term.step_index.cpu().numpy()
                mj.mjr_rectangle(mj.MjrRect(0, 0, width, height), 0.025, 0.043, 0.06, 1)
                pad, top = 5, 32
                tile_w, tile_h = (width - pad) // 4 - pad, (height - top - pad) // 4 - pad
                for i in range(N):
                    row, column = divmod(i, 4)
                    tile = mj.MjrRect(
                        pad + column * (tile_w + pad),
                        height - top - pad - tile_h - row * (tile_h + pad),
                        tile_w,
                        tile_h,
                    )
                    data.qpos[:] = pose[i]
                    mj.mj_kinematics(model, data)
                    mj.mj_comPos(model, data)
                    mj.mj_camlight(model, data)
                    cam.lookat[:] = [positions[i, 0], positions[i, 1], 0.30]
                    mj.mjv_updateScene(
                        model, data, option, None, cam, mj.mjtCatBit.mjCAT_ALL, scene
                    )
                    heading = origin[i, 2]
                    cos, sin = math.cos(heading), math.sin(heading)
                    rotation = np.array([[cos, -sin], [sin, cos]])
                    local_path = reference[i, ::15, :2]
                    world_path = local_path @ rotation.T + origin[i, :2]
                    path = np.column_stack((world_path, np.full(len(world_path), 0.024)))
                    draw_reference(scene, path)
                    mj.mjr_render(tile, scene, ctx)
                    local = (positions[i, :2] - origin[i, :2]) @ rotation
                    if not traces[i] or np.linalg.norm(local - traces[i][-1]) > 0.004:
                        traces[i].append(local.copy())
                    stick = turn_direction(reference[i], feedforward[i], index[i], cuts)
                    topdown(
                        reference[i, ::8],
                        list(traces[i]),
                        local,
                        headings[i] - heading,
                        stick,
                        tile,
                    )
                    direction = (
                        "STOP"
                        if np.linalg.norm(stick) < 0.001
                        else ("REVERSE" if stick[1] < 0 else "FORWARD")
                    )
                    overlay(
                        ctx,
                        tile,
                        (
                            f"{i + 1:02d}  {direction}  {index[i] * env.step_dt:4.1f}s\n"
                            f"Speed {commands[i, 0]:+.2f} / {speed[i]:+.2f} m/s\n"
                            f"Steer {math.degrees(commands[i, 1]):+.0f} deg"
                        ),
                    )

                status = "Paused" if requested["paused"] else "Running"
                banner = mj.MjrRect(5, height - top, width - 10, top)
                overlay(
                    ctx,
                    banner,
                    (
                        f"PUKY JOYSTICK   |   {status}   |   Cyan: reference   Amber: actual   "
                        "|   Speed: target / actual   |   Space: pause   R: resample   Esc: close"
                    ),
                )
                frames += 1
                if frames == 1:
                    print(f"Rendered 4x4 tiled window at {width}x{height}", flush=True)
                glfw.swap_buffers(window)
                next_frame = max(next_frame + 1 / 24, time.monotonic())
                glfw.wait_events_timeout(max(0.001, next_frame - time.monotonic()))
    finally:
        ctx.free()
        glfw.destroy_window(window)
        glfw.terminate()
        env.close()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
