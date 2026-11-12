"""booster_deploy task "laai_sit_hold": hold the K1's solved seated pose on the Puky (no policy).

For the first real-robot tests of the seated pose (experiments/sit/README.md has the full
procedure). Nothing is learned and nothing balances: every joint is held at the seated pose
from models/k1_on_bike/puky_bike.json with Booster's k1_walk PD gains, the same gains the
simulation holds it with. The robot stays up only through its feet and the bike's wheels.

Source of truth: experiments/sit/deploy/laai_sit.py in the project repository. robot/install.sh
copies it to booster_deploy/tasks/locomotion/robots/k1/ and the seated poses to
.../k1/models/seated_poses/<bike>.json.

Sequence on the robot (robot in damping mode, placed on the bike by hand, bike held upright):
  * X: booster_deploy's "standing" preparation slides all joints from where they are to the
    seated pose over 1 s (this task's prepare pose is the seated pose, with the k1_walk gains,
    not Booster's stiff standing gains). Booster's walking mode is never entered.
  * A: this task starts and holds the seated pose. Its targets slide from the last command
    over HANDOVER_S, so starting it never jumps.
  * Any stop or exit: damping mode (limp). Someone must hold the robot.

Safety stops (each prints the reason and saves the last LOG_SECONDS to logs/stop_<time>.npz):
  * the trunk tilts more than MAX_TILT_CHANGE_DEG away from its seated orientation (the bike
    is falling over, or the robot is sliding off);
  * any joint moves faster than MAX_JOINT_VEL rad/s (holding a pose should be nearly still);
  * any joint's PD torque stays above TORQUE_FRACTION of its limit for TORQUE_SECONDS (the
    joint is fighting something, for example hand clips that sit elsewhere than assumed);
  * any joint stays more than MAX_TRACKING_ERROR rad from its target for TRACKING_SECONDS.
"""

import json
import math
import os
import time
from collections import deque

import numpy as np
import torch
from booster_deploy.controllers.base_controller import Policy
from booster_deploy.controllers.controller_cfg import (
    BoosterRobotControllerCfg,
    PolicyCfg,
    PrepareStateCfg,
)
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from . import K1WalkControllerCfg

BIKE = "puky_bike"  # the bike the robot sits on (its pose file: models/seated_poses/<bike>.json)
MAX_TILT_CHANGE_DEG = 15.0  # trunk tilt away from the seated orientation
MAX_JOINT_VEL = 4.0  # rad/s
TORQUE_FRACTION = 0.9  # of each joint's effort limit
TORQUE_SECONDS = 0.5
MAX_TRACKING_ERROR = 0.3  # rad
TRACKING_SECONDS = 1.0
HANDOVER_S = 1.0  # seconds to slide from the previous command to the seated pose
LOG_SECONDS = 5.0  # history saved when the task stops

_HERE = os.path.dirname(os.path.abspath(__file__))
# Installed: next to this file in booster_deploy. In the project repository: models/k1_on_bike/.
_POSE_DIRS = (
    os.path.join(_HERE, "models", "seated_poses"),
    os.path.join(_HERE, "..", "..", "..", "models", "k1_on_bike"),
)


def pose_file(bike: str = BIKE) -> str | None:
    for folder in _POSE_DIRS:
        path = os.path.join(folder, f"{bike}.json")
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def seated_targets(joint_names, fallback, bike: str = BIKE) -> tuple[list[float], list[float]]:
    """Seated joint angles in `joint_names` order (joints the pose does not set, the head,
    keep `fallback`), and the trunk's seated orientation (w, x, y, z) relative to an upright
    bike. Without a pose file: (fallback, upright); the policy then refuses to start."""
    path = pose_file(bike)
    if path is None:
        return list(fallback), [1.0, 0.0, 0.0, 0.0]
    with open(path) as f:
        pose = json.load(f)
    joints = pose["joints"]
    return [float(joints.get(n, d)) for n, d in zip(joint_names, fallback)], pose["trunk_quat"]


class SitHoldPolicy(Policy):
    """Hold the seated pose; stop on tilt, speed, torque or tracking limits."""

    def __init__(self, cfg, controller):
        torch.set_num_threads(1)
        super().__init__(cfg, controller)
        if pose_file(cfg.bike) is None:
            raise FileNotFoundError(
                f"No seated pose for '{cfg.bike}' in {_POSE_DIRS[0]}. "
                "Install it with robot/install.sh sit."
            )
        self.robot = controller.robot
        self.device = cfg.device
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)  # the seated pose
        _, trunk_quat = seated_targets(
            self.robot.cfg.joint_names, self.robot.cfg.default_joint_pos, cfg.bike
        )
        # Gravity as the trunk should feel it when seated on an upright bike.
        self._seated_gravity = lab_math.quat_apply_inverse(
            torch.tensor(trunk_quat, dtype=torch.float32),
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        ).to(self.device)
        n = self.robot.num_joints
        self.last_action = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._dt = controller.cfg.policy_dt
        self._handover_steps = max(1, round(HANDOVER_S / self._dt))
        self._torque_steps = max(1, round(TORQUE_SECONDS / self._dt))
        self._tracking_steps = max(1, round(TRACKING_SECONDS / self._dt))
        self._history = deque(maxlen=max(1, round(LOG_SECONDS / self._dt)))
        self.reset()

    # Interface used by robot/dry_run.py (same names as booster_deploy's LocomotionPolicy).
    def _action_to_targets(self, action: torch.Tensor) -> torch.Tensor:
        return self.default_joint_pos + action

    def _previous_targets(self) -> torch.Tensor:
        """The last command the robot received (booster_deploy's prepare slide ends on the
        seated pose); in simulation, where there is none, the current joint angles."""
        portal = getattr(self.controller, "portal", None)
        if portal is not None:
            try:
                q = torch.tensor([float(m.q) for m in portal.motor_cmd], dtype=torch.float32)
                if q.numel() == self.default_joint_pos.numel() and torch.isfinite(q).all():
                    return q.to(self.device)
            except Exception:  # noqa: BLE001, S110  (fall back to the measured angles)
                pass
        return self.robot.data.joint_pos.to(self.device).clone()

    def reset(self) -> None:
        self._step = 0
        self._stopped = False
        self._over_torque = torch.zeros(self.robot.num_joints, dtype=torch.int32)
        self._off_target = torch.zeros(self.robot.num_joints, dtype=torch.int32)
        self._history.clear()
        self._handover_start = None
        self.stop_reason = None

    def _gravity(self) -> torch.Tensor:
        return lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w.to(self.device),
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device),
        )

    def tilt_change_deg(self) -> float:
        cos = float(torch.dot(self._gravity(), self._seated_gravity))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    def pd_torque(self, targets: torch.Tensor) -> torch.Tensor:
        d = self.robot.data
        q, dq = d.joint_pos.to(self.device), d.joint_vel.to(self.device)
        kp, kd = (
            self.robot.joint_stiffness.to(self.device),
            self.robot.joint_damping.to(self.device),
        )
        return kp * (targets - q) - kd * dq

    def _trip(self, reason: str) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.stop_reason = reason
        print(f"\n[SAFETY] {reason} -> stopping (robot enters the exit mode, damping).")
        self.controller.stop()
        self._save_history(reason)

    def _save_history(self, reason: str) -> None:
        try:
            t, q, dq, tgt, grav = (np.array(x) for x in zip(*self._history))
            os.makedirs("logs", exist_ok=True)
            stamp = time.strftime("stop_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            path = os.path.join("logs", f"{stamp}.npz")
            meta = {
                "task": "laai_sit_hold",
                "bike": self.cfg.bike,
                "reason": reason,
                "joint_names": list(self.robot.cfg.joint_names),
                "policy_dt": self._dt,
                "seated_pose": self.default_joint_pos.cpu().tolist(),
            }
            np.savez(
                path,
                meta=json.dumps(meta),
                t=t,
                joint_pos=q,
                joint_vel=dq,
                target=tgt,
                gravity=grav,
            )
            print(f"  saved the last {t[-1] - t[0] + self._dt:.1f} s to {os.path.abspath(path)}")
        except Exception as exc:  # noqa: BLE001  (diagnostics must never block the stop)
            print(f"  (could not write stop details: {exc})")

    def inference(self) -> torch.Tensor:
        if self._handover_start is None:
            self._handover_start = self._previous_targets()
        self._step += 1
        targets = self.default_joint_pos.clone()
        if self._step < self._handover_steps:
            alpha = self._step / self._handover_steps
            targets = self._handover_start + alpha * (targets - self._handover_start)
        self.last_action = targets - self.default_joint_pos

        d = self.robot.data
        self._history.append(
            (
                self._step * self._dt,
                d.joint_pos.detach().cpu().numpy().copy(),
                d.joint_vel.detach().cpu().numpy().copy(),
                targets.detach().cpu().numpy().copy(),
                self._gravity().detach().cpu().numpy().copy(),
            )
        )
        names = self.robot.cfg.joint_names

        tilt = self.tilt_change_deg()
        if tilt > MAX_TILT_CHANGE_DEG:
            self._trip(
                f"trunk tilted {tilt:.0f} deg from its seated orientation > {MAX_TILT_CHANGE_DEG}"
            )

        speed = d.joint_vel.abs()
        if float(speed.max()) > MAX_JOINT_VEL:
            j = int(speed.argmax())
            self._trip(f"joint speed {float(speed[j]):.1f} rad/s at {names[j]} > {MAX_JOINT_VEL}")

        limit = self.robot.effort_limit.to(self.device)
        over = (self.pd_torque(targets).abs() > TORQUE_FRACTION * limit).cpu()
        self._over_torque = torch.where(
            over, self._over_torque + 1, torch.zeros_like(self._over_torque)
        )
        if int(self._over_torque.max()) >= self._torque_steps:
            j = int(self._over_torque.argmax())
            self._trip(
                f"{names[j]} at over {TORQUE_FRACTION:.0%} of its torque limit for {TORQUE_SECONDS} s "
                "(fighting the clips or the bike?)"
            )

        off = ((targets - d.joint_pos.to(self.device)).abs() > MAX_TRACKING_ERROR).cpu()
        self._off_target = torch.where(
            off, self._off_target + 1, torch.zeros_like(self._off_target)
        )
        if int(self._off_target.max()) >= self._tracking_steps:
            j = int(self._off_target.argmax())
            self._trip(
                f"{names[j]} more than {MAX_TRACKING_ERROR} rad from its target for {TRACKING_SECONDS} s"
            )
        return targets


@configclass
class LaaiSitHoldPolicyCfg(PolicyCfg):
    constructor = SitHoldPolicy
    bike: str = BIKE
    # The pose file plays the part of the checkpoint (robot/dry_run.py prints it).
    checkpoint_path: str = f"models/seated_poses/{BIKE}.json"
    policy_joint_names: list = None  # set below: all 22 joints are held


@configclass
class LaaiSitHoldTaskCfg(K1WalkControllerCfg):
    bike: str = BIKE  # which bike's seated pose to hold (registered task: the Puky)

    def __post_init__(self):
        super().__post_init__()
        names = list(self.robot.joint_names)
        seated, _ = seated_targets(names, self.robot.default_joint_pos, self.bike)
        self.robot = self.robot.replace(
            default_joint_pos=seated,
            # X slides to the seated pose with the k1_walk gains, never via Booster's walking.
            prepare_mode="standing",
            prepare_state=PrepareStateCfg(
                stiffness=list(self.robot.joint_stiffness),
                damping=list(self.robot.joint_damping),
                joint_pos=seated,
            ),
        )
        self.vel_command = None
        self.booster = BoosterRobotControllerCfg(exit_mode="damping")
        self.policy = LaaiSitHoldPolicyCfg(
            policy_joint_names=names,
            bike=self.bike,
            checkpoint_path=f"models/seated_poses/{self.bike}.json",
        )


register_task("laai_sit_hold", LaaiSitHoldTaskCfg())
