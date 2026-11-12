"""booster_deploy tasks "laai_walk" and "laai_walk_ankle_kd15": run our phase 1 walking policy
(trained with experiments/walk/task.py),
with extra safety for first tests on the real robot.

Source of truth: experiments/walk/deploy/laai_walk.py in the project repository. robot/install.sh copies it
to booster_deploy/tasks/locomotion/robots/k1/ and the policy to .../k1/models/laai_walk.pt.

Commands: docs/commands.md (install, Booster's MuJoCo, robot).

Robot settings (default pose, PD gains, effort limits, joint order) are Booster's own k1_walk
settings. The exported model already contains the observation normalization; no action filter.

Safety additions (on top of booster_deploy's own 60-degree fall stop):
  * Speed limits: even a full stick only asks for 0.5 m/s forward, 0.2 m/s back/sideways,
    0.6 rad/s turning (all well inside what the policy was trained on).
  * Smooth handover: for the first HANDOVER_S seconds the joint targets slide from the last
    targets the robot received (Booster's standing policy after X) to this policy's targets,
    instead of jumping there in one step (up to 0.5 rad at the elbows, 0.15 rad at the knees).
  * Shake watchdog: stops the policy if
      - any joint moves faster than MAX_JOINT_VEL rad/s, or
      - the policy's commands jump by more than MAX_ACTION_JUMP for JUMP_STEPS steps in a row, or
      - the average command jump over the last SHAKE_WINDOW steps exceeds MAX_MEAN_JUMP
        (sustained back-and-forth = shaking; catches the run 1 failure 2-3 s earlier), or
      - the body tilts more than MAX_TILT_DEG.
    Every stop prints which joint was fastest, its last 0.2 s of speed and position, and saves the
    last LOG_SECONDS of sensor and policy data to logs/stop_<time>.npz. This also covers
    booster_deploy's own fall stop.
    After a stop the robot enters the exit mode: Booster's walking mode by default, or damping
    (limp) with --exit-mode damping.
    Thresholds from booster_deploy MuJoCo: v2 walking at all command limits peaks at 7.8 rad/s,
    a 0.72 single jump and a 0.55 average jump; run 1 shaking averages 0.9-2.1.
"""

import json
import math
import os
import time
from collections import deque

import numpy as np
import torch
from booster_deploy.controllers.controller_cfg import VelocityCommandCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...locomotion import K1LocomotionPolicyCfg, LocomotionPolicy
from . import K1WalkControllerCfg

MAX_JOINT_VEL = 15.0  # rad/s
MAX_ACTION_JUMP = 2.5  # norm of the change in the 20 actions between two steps
JUMP_STEPS = 3  # that many steps in a row
SHAKE_WINDOW = 10  # steps (0.2 s at 50 Hz)
MAX_MEAN_JUMP = 0.8  # average action jump over SHAKE_WINDOW steps
MAX_TILT_DEG = 35.0  # body tilt from upright
HANDOVER_S = 0.5  # seconds to slide from the previous targets to this policy's targets
LOG_SECONDS = 5.0  # history saved when the policy is stopped


class SafeLocomotionPolicy(LocomotionPolicy):
    def __init__(self, cfg, controller):
        # One CPU thread: on the robot, PyTorch's default multi-threading made each 20 ms
        # policy step take 13 ms (6.7 ms with one thread, measured with scripts/dry_run.py).
        torch.set_num_threads(1)
        super().__init__(cfg, controller)
        self._prev_action = None
        self._jump_count = 0
        self._jumps = deque(maxlen=SHAKE_WINDOW)
        self._stopped = False
        self._handover_steps = max(1, round(HANDOVER_S / controller.cfg.policy_dt))
        self._handover_start = None
        self._step = 0
        self._dt = controller.cfg.policy_dt
        self._history = deque(maxlen=max(1, round(LOG_SECONDS / self._dt)))

    def _previous_targets(self) -> torch.Tensor:
        """Joint targets the robot was holding before this policy started.

        On the robot this is the last command published (Booster's standing policy after X,
        or the prepare pose); it lives in the portal shared with the previous controller.
        In MuJoCo there is no previous controller, so use the default pose.
        """
        portal = getattr(self.controller, "portal", None)
        if portal is not None:
            try:
                q = torch.tensor(
                    [float(m.q) for m in portal.motor_cmd], dtype=torch.float32, device=self.device
                )
                if q.numel() == self.default_joint_pos.numel() and torch.isfinite(q).all():
                    return q
            except Exception:  # noqa: BLE001, S110  (fall back to the default pose below)
                pass
        return self.default_joint_pos.clone()

    def reset(self) -> None:
        super().reset()
        self._prev_action = None
        self._jump_count = 0
        self._jumps.clear()
        self._stopped = False
        self._step = 0
        self._history.clear()
        self._handover_start = self._previous_targets()

    def _record(self, action: torch.Tensor, targets: torch.Tensor, gravity: torch.Tensor) -> None:
        d = self.robot.data
        cmd = self.controller.vel_command
        self._history.append(
            (
                self._step * self._dt,
                d.joint_pos.detach().cpu().numpy().copy(),
                d.joint_vel.detach().cpu().numpy().copy(),
                action.detach().cpu().numpy().copy(),
                targets.detach().cpu().numpy().copy(),
                gravity.detach().cpu().numpy().copy(),
                d.root_ang_vel_b.detach().cpu().numpy().copy(),
                np.array(
                    [cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw]
                    if cmd is not None
                    else [0, 0, 0],
                    dtype=np.float32,
                ),
            )
        )

    def _report_stop(self, reason: str) -> None:
        """Print what the joints did just before the stop and save the recent history."""
        try:
            names = list(self.robot.cfg.joint_names)
            t, q, dq, act, tgt, grav, gyro, cmd = (np.array(x) for x in zip(*self._history))
            last = max(1, round(0.2 / self._dt))
            peak = np.abs(dq[-max(1, round(1.0 / self._dt)) :]).max(0)
            j = int(np.argmax(np.abs(dq[-1])))
            order = np.argsort(-peak)[:3]
            print(f"  time since start: {t[-1]:.2f} s, command {np.round(cmd[-1], 2).tolist()}")
            print(f"  fastest joint now: {names[j]} at {dq[-1, j]:+.1f} rad/s")
            print(
                f"  {names[j]} speed, last {last * self._dt:.1f} s (rad/s):    "
                + " ".join(f"{v:+.1f}" for v in dq[-last:, j])
            )
            print(
                f"  {names[j]} position, last {last * self._dt:.1f} s (rad): "
                + " ".join(f"{v:+.2f}" for v in q[-last:, j])
            )
            print(
                "  fastest joints in the last 1 s: "
                + ", ".join(f"{names[i]} {peak[i]:.1f}" for i in order)
            )
            os.makedirs("logs", exist_ok=True)
            path = os.path.join(
                "logs",
                time.strftime("stop_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.npz",
            )
            meta = {
                "reason": reason,
                "joint_names": names,
                "policy_dt": self._dt,
                "default_joint_pos": self.default_joint_pos.cpu().tolist(),
            }
            np.savez(
                path,
                meta=json.dumps(meta),
                t=t,
                joint_pos=q,
                joint_vel=dq,
                action=act,
                target=tgt,
                gravity=grav,
                gyro=gyro,
                command=cmd,
            )
            print(f"  saved the last {t[-1] - t[0] + self._dt:.1f} s to {os.path.abspath(path)}")
        except Exception as exc:  # noqa: BLE001  (diagnostics must never block the stop)
            print(f"  (could not write stop details: {exc})")

    def _trip(self, reason: str) -> None:
        if not self._stopped:
            self._stopped = True
            print(f"\n[SAFETY] {reason} -> stopping the policy (robot enters the exit mode).")
            self.controller.stop()
            self._report_stop(reason)

    def inference(self) -> torch.Tensor:
        was_running = self.controller.is_running
        targets = super().inference()
        action = self.last_action.clone()
        gravity = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device),
        )
        self._record(action, targets, gravity)
        if was_running and not self.controller.is_running and not self._stopped:
            # booster_deploy's own fall check (in compute_observation) stopped the controller.
            self._stopped = True
            self._report_stop("booster_deploy fall check (body tilted more than about 60 deg)")

        joint_speed = self.robot.data.joint_vel[self.real2sim_joint_map].abs()
        joint_vel = joint_speed.max().item()
        if joint_vel > MAX_JOINT_VEL:
            name = self.robot.cfg.joint_names[
                int(self.real2sim_joint_map[int(joint_speed.argmax())])
            ]
            self._trip(f"joint speed {joint_vel:.1f} rad/s at {name} > {MAX_JOINT_VEL}")

        if self._prev_action is not None:
            jump = (action - self._prev_action).norm().item()
            self._jump_count = self._jump_count + 1 if jump > MAX_ACTION_JUMP else 0
            if self._jump_count >= JUMP_STEPS:
                self._trip(f"action jumps > {MAX_ACTION_JUMP} for {JUMP_STEPS} steps (shaking)")
            self._jumps.append(jump)
            if len(self._jumps) == SHAKE_WINDOW:
                mean_jump = sum(self._jumps) / SHAKE_WINDOW
                if mean_jump > MAX_MEAN_JUMP:
                    self._trip(
                        f"average action jump {mean_jump:.2f} over "
                        f"{SHAKE_WINDOW * self.controller.cfg.policy_dt:.1f} s > {MAX_MEAN_JUMP} (shaking)"
                    )
        self._prev_action = action

        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gravity[2].item()))))
        if tilt > MAX_TILT_DEG:
            self._trip(f"body tilt {tilt:.0f} deg > {MAX_TILT_DEG}")

        # Smooth handover: slide from the previous targets to the policy's targets.
        self._step += 1
        if self._handover_start is not None and self._step < self._handover_steps:
            alpha = self._step / self._handover_steps
            targets = self._handover_start + alpha * (targets - self._handover_start)
        return targets


@configclass
class LaaiWalkPolicyCfg(K1LocomotionPolicyCfg):
    constructor = SafeLocomotionPolicy


@configclass
class LaaiWalkTaskCfg(K1WalkControllerCfg):
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_forward_max=0.5,
        vx_backward_max=0.2,
        vy_max=0.2,
        vyaw_max=0.6,
    )

    def __post_init__(self):
        super().__post_init__()
        self.policy = LaaiWalkPolicyCfg(
            policy_joint_names=self.policy.policy_joint_names,
            # Relative to this file's folder (tasks/locomotion/robots/k1/).
            checkpoint_path="models/laai_walk.pt",
            actor_obs_history_length=10,
            history_init="repeat",
            obs_dof_vel_scale=0.1,
            action_scale=0.25,
            use_actor_module=False,
            action_filter=1.0,
            arm_action_fix_joint_name=None,
            arm_action_fix_offset=0.0,
        )


register_task("laai_walk", LaaiWalkTaskCfg())


# Variant with more ankle damping. Robot test 2026-11-11: with kd 1.0 the ankle rolls built up a
# ~13 Hz oscillation within 2 s of walking (policy and ankle feeding each other, joint swinging
# 3x further than commanded), while Booster's k1_walk walked cleanly in the same test.
# 1.5 stays inside run 2's training randomization (kd x0.6 to x1.6 of the base 1.0).
ANKLE_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
ANKLE_KD = 1.5


@configclass
class LaaiWalkAnkleKdTaskCfg(LaaiWalkTaskCfg):
    def __post_init__(self):
        super().__post_init__()
        kd = list(self.robot.joint_damping)
        for name in ANKLE_JOINTS:
            kd[self.robot.joint_names.index(name)] = ANKLE_KD
        self.robot = self.robot.replace(joint_damping=kd)


register_task("laai_walk_ankle_kd15", LaaiWalkAnkleKdTaskCfg())
