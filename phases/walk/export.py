"""Export a trained K1 walking policy for Booster's deploy tool, check it, and record a video.

Linux GPU cluster node or NVIDIA host (from the repository root):
    uv run --group train python -m phases.walk.export --checkpoint logs/rsl_rl/laai_walk/<run>/model_3999.pt
    MUJOCO_GL=egl uv run --group train python -m phases.walk.export --checkpoint ... --video walk.mp4
    uv run --group train python -m phases.walk.export --task LAAI-K1-Walk-Flat-NoSelfCollision \
        --checkpoint ...   # run 3; same policy interface, its own simulation settings

Then keep policy.pt as phases/walk/runs/runN/policy.pt and install it with deploy/install.sh walk.

Writes policy.pt (TorchScript). booster_deploy feeds it 690 numbers (10 frames x 69, oldest
first, joints in booster_deploy's policy order) and expects 20 actions in that same joint order.
mjlab orders observations per term and joints in model order, so the exported module reorders
both ways. The script verifies this by computing the observation the way booster_deploy does,
from the simulator state, and comparing the actions with mjlab's own.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from torch import nn

from phases.walk import task as K

FRAME = 3 + 3 + 3 + 20 + 20 + 20  # 69 numbers per frame, booster_deploy layout


class DeployPolicy(nn.Module):
    """booster_deploy layout in -> mjlab policy -> booster_deploy joint order out."""

    def __init__(self, core: nn.Module, in_index: torch.Tensor, out_index: torch.Tensor):
        super().__init__()
        self.core = core
        self.register_buffer("in_index", in_index)
        self.register_buffer("out_index", out_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 690)
        actions = self.core(x[:, self.in_index])
        return actions[:, self.out_index].reshape(-1)


def build_index_maps(mjlab_joints: list[str]):
    """in_index[i] = which deploy input feeds mjlab input i; out_index[j] = mjlab action for deploy joint j."""
    deploy_pos = {name: i for i, name in enumerate(K.POLICY_JOINTS)}
    to_deploy = [deploy_pos[n] for n in mjlab_joints]  # mjlab joint k -> deploy slot
    # mjlab actor layout: per term, H frames (oldest first), each frame = term values.
    terms = [
        ("ang", 3, 0),
        ("grav", 3, 3),
        ("cmd", 3, 6),
        ("pos", 20, 9),
        ("vel", 20, 29),
        ("act", 20, 49),
    ]
    in_index = []
    for name, size, offset in terms:
        for frame in range(K.HISTORY):
            for k in range(size):
                slot = to_deploy[k] if size == 20 else k
                in_index.append(frame * FRAME + offset + slot)
    mjlab_pos = {n: k for k, n in enumerate(mjlab_joints)}
    out_index = [mjlab_pos[n] for n in K.POLICY_JOINTS]
    return torch.tensor(in_index, dtype=torch.long), torch.tensor(out_index, dtype=torch.long)


def make_env(task: str, play: bool, render: bool, device: str):
    cfg = load_env_cfg(task, play=play) if play else load_env_cfg(task)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    return ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array" if render else None)


def deploy_frame(env, mjlab_joints, last_action_mjlab):
    """Build one 69-number frame exactly like booster_deploy's LocomotionPolicy.compute_observation."""
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    ids = [names.index(n) for n in K.POLICY_JOINTS]
    q = robot.data.joint_pos_biased[0, ids]
    qd = robot.data.joint_vel[0, ids]
    q0 = robot.data.default_joint_pos[0, ids]
    ang = env.scene["robot/imu_ang_vel"].data[0]
    grav = robot.data.projected_gravity_b[0]
    cmd = env.command_manager.get_command("twist")[0]
    mj_pos = {n: k for k, n in enumerate(mjlab_joints)}
    act = last_action_mjlab[[mj_pos[n] for n in K.POLICY_JOINTS]]
    return torch.cat([ang, grav, cmd, q - q0, qd * K.JOINT_VEL_SCALE, act])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--task",
        default=K.TASK_ID,
        choices=(K.TASK_ID, K.RUN3_TASK_ID),
        help="walk task the checkpoint was trained with",
    )
    ap.add_argument("--out", default="policy.pt")
    ap.add_argument("--video", default=None, help="optional mp4 file to record the policy walking")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    env = make_env(args.task, play=True, render=args.video is not None, device=args.device)
    agent_cfg = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)

    core = runner.alg.get_policy().as_jit().to("cpu").eval()
    mjlab_joints = list(env.action_manager.get_term("joint_pos").target_names)
    in_index, out_index = build_index_maps(mjlab_joints)
    deploy_model = torch.jit.script(DeployPolicy(core, in_index, out_index))
    deploy_model.save(args.out)
    print(f"Saved {args.out}")

    # --- Run the policy, check the deploy-format model gives the same actions, record video ---
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.get_observations()
    cmd_term = env.command_manager.get_term("twist")
    history: deque = deque(maxlen=K.HISTORY)
    last_action = torch.zeros(20, device=args.device)
    frames, max_diff, fell = [], 0.0, False
    steps = int(args.seconds / env.step_dt)

    def set_command(t):
        # Scripted commands for the test: stand, walk forward, curve, walk back, stand.
        phase = t * env.step_dt / args.seconds
        vx, wz = (
            (0.0, 0.0)
            if phase < 0.1
            else (0.5, 0.0)
            if phase < 0.5
            else (0.3, 0.6)
            if phase < 0.7
            else (-0.2, 0.0)
            if phase < 0.9
            else (0.0, 0.0)
        )
        cmd_term.vel_command_b[:] = torch.tensor(
            [vx, 0.0, wz], device=cmd_term.vel_command_b.device
        )
        cmd_term.time_left[:] = 1e9
        for flag in ("is_heading_env", "is_standing_env", "is_world_env"):
            getattr(cmd_term, flag)[:] = False

    def push_frame():
        frame = deploy_frame(env, mjlab_joints, last_action).cpu()
        if not history:
            history.extend([frame] * K.HISTORY)  # booster_deploy's "repeat" history start
        else:
            history.append(frame)

    obs, _ = wrapped.reset()
    history.clear()
    push_frame()
    set_command(0)
    with torch.no_grad():
        for t in range(steps):
            action = policy(obs)
            deploy_action = deploy_model(torch.stack(list(history)).flatten())
            max_diff = max(
                max_diff, float((deploy_action - action[0].cpu()[out_index]).abs().max())
            )
            obs, _, dones, _ = wrapped.step(action)
            last_action = action[0].detach()
            if bool(dones[0]) and t < steps - 1:
                fell = True
                history.clear()
                last_action = torch.zeros(20, device=args.device)
            push_frame()  # same moment mjlab computes its observation
            set_command(t + 1)  # command for the next step
            if args.video:
                frames.append(env.render())

    print(
        f"Deploy-format check: max action difference over the run = {max_diff:.2e} "
        f"({'OK' if max_diff < 1e-4 else 'MISMATCH - do not deploy'})"
    )
    print("Robot fell during the test run!" if fell else "Robot stayed up for the whole test run.")
    if args.video:
        import imageio

        imageio.mimsave(args.video, [np.asarray(f) for f in frames], fps=round(1 / env.step_dt))
        print(f"Saved video {args.video}")


if __name__ == "__main__":
    main()
