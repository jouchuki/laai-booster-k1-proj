"""booster_deploy tasks for the K1 seated on the Puky (phase 3):

  * "laai_sit_hold": hold the solved seated pose, no policy. For the first robot tests of the
    pose itself. Nothing is learned and nothing balances: every joint is held at the seated pose
    from models/k1_on_bike/puky_bike.json with Booster's k1_walk PD gains, the same gains the
    simulation holds it with. The robot stays up only through its feet and the bike's wheels.
  * "laai_sit": run the trained sit policy (experiments/sit/task.py, exported with
    experiments/sit/export.py) around the same seated pose. Interface as the walk policy: 690
    inputs (10 frames of 69, velocity command always zero), 20 actions, target = seated pose +
    0.25 x action.
  * "laai_sit_ankle_kd15": "laai_sit" with ankle damping 1.5 instead of 1.0 (see the note at
    its registration below).

Source of truth: experiments/sit/deploy/laai_sit.py in the project repository. robot/install.sh
copies it to booster_deploy/tasks/locomotion/robots/k1/, the seated poses to
.../k1/models/seated_poses/<bike>.json and the sit policy to .../k1/models/laai_sit.pt.

Sequence on the robot (robot in damping mode, placed on the bike by hand, bike held upright):
  * X: booster_deploy's "standing" preparation slides all joints from where they are to the
    seated pose over 1 s (the prepare pose is the seated pose, with the k1_walk gains, not
    Booster's stiff standing gains). Booster's walking mode is never entered.
  * A (keyboard r), within a few seconds of X: the task starts. Its targets slide from the last
    command over HANDOVER_S, so starting it never jumps. The safety stops below run only from
    here on.
  * Any stop or exit: damping mode (limp). Someone must hold the robot.

Safety stops, in both tasks (each prints the reason and saves the last LOG_SECONDS to
logs/stop_<time>.npz):
  * the trunk tilts more than MAX_TILT_CHANGE_DEG away from its seated orientation (the bike
    is falling over, or the robot is sliding off);
  * any joint moves faster than MAX_JOINT_VEL rad/s;
  * any joint's PD torque stays above TORQUE_FRACTION of its limit for TORQUE_SECONDS (the
    joint is fighting something, for example hand clips that sit elsewhere than assumed);
  * any joint stays more than MAX_TRACKING_ERROR rad from its target for TRACKING_SECONDS.
"laai_sit" also stops when the policy shakes: action jumps above MAX_ACTION_JUMP for JUMP_STEPS
steps in a row, or an average jump above MAX_MEAN_JUMP over SHAKE_WINDOW steps (as the walk
task).
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

from ...locomotion import K1LocomotionPolicyCfg, LocomotionPolicy
from . import K1WalkControllerCfg

BIKE = "puky_bike"  # the bike the robot sits on (its pose file: models/seated_poses/<bike>.json)
# Relative to this file's folder; robot/install.sh sit puts the newest run's policy.pt there.
POLICY_FILE = "models/laai_sit.pt"
MAX_TILT_CHANGE_DEG = 15.0  # trunk tilt away from the seated orientation
MAX_JOINT_VEL = 4.0  # rad/s; seated, with the bike held, the joints should barely move
TORQUE_FRACTION = 0.9  # of each joint's effort limit
TORQUE_SECONDS = 0.5
MAX_TRACKING_ERROR = 0.3  # rad
TRACKING_SECONDS = 1.0
MAX_ACTION_JUMP = 2.5  # norm of the change in the 20 actions between two steps (laai_sit)
JUMP_STEPS = 3  # that many steps in a row
SHAKE_WINDOW = 10  # steps (0.2 s at 50 Hz)
MAX_MEAN_JUMP = 0.8  # average action jump over SHAKE_WINDOW steps
HANDOVER_S = 1.0  # seconds to slide from the previous command to the task's targets
ANKLE_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
ANKLE_KD = 1.5  # laai_sit_ankle_kd15 (k1_walk: 1.0), as laai_walk_ankle_kd15
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


def _require_pose(bike: str) -> None:
    if pose_file(bike) is None:
        raise FileNotFoundError(
            f"No seated pose for '{bike}' in {_POSE_DIRS[0]}. Install it with robot/install.sh sit."
        )


class SeatedSafety:
    """Safety stops shared by both sit tasks. Call `_init_safety` after the policy's own
    __init__, `_reset_safety` from reset() and `_check_safety` once per step."""

    task_name = ""

    def _init_safety(self, controller) -> None:
        self._dt = controller.cfg.policy_dt
        _, trunk_quat = seated_targets(
            self.robot.cfg.joint_names, self.robot.cfg.default_joint_pos, self.cfg.bike
        )
        # Gravity as the trunk should feel it when seated on an upright bike.
        self._seated_gravity = lab_math.quat_apply_inverse(
            torch.tensor(trunk_quat, dtype=torch.float32),
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        ).to(self.device)
        self._handover_steps = max(1, round(HANDOVER_S / self._dt))
        self._torque_steps = max(1, round(TORQUE_SECONDS / self._dt))
        self._tracking_steps = max(1, round(TRACKING_SECONDS / self._dt))
        self._history = deque(maxlen=max(1, round(LOG_SECONDS / self._dt)))
        self._reset_safety()

    def _reset_safety(self) -> None:
        self._step = 0
        self._stopped = False
        self._over_torque = torch.zeros(self.robot.num_joints, dtype=torch.int32)
        self._off_target = torch.zeros(self.robot.num_joints, dtype=torch.int32)
        self._history.clear()
        self._handover_start = None
        self.stop_reason = None

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

    def _handover(self, targets: torch.Tensor) -> torch.Tensor:
        """Slide from the previous command to `targets` over the first HANDOVER_S."""
        if self._handover_start is None:
            self._handover_start = self._previous_targets()
        self._step += 1
        if self._step < self._handover_steps:
            alpha = self._step / self._handover_steps
            return self._handover_start + alpha * (targets - self._handover_start)
        return targets

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
            t, q, dq, act, tgt, grav = (np.array(x) for x in zip(*self._history))
            os.makedirs("logs", exist_ok=True)
            stamp = time.strftime("stop_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            path = os.path.join("logs", f"{stamp}.npz")
            meta = {
                "task": self.task_name,
                "bike": self.cfg.bike,
                "reason": reason,
                "joint_names": list(self.robot.cfg.joint_names),
                "policy_joint_names": list(self.cfg.policy_joint_names),
                "policy_dt": self._dt,
                "seated_pose": self.default_joint_pos.cpu().tolist(),
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
            )
            print(f"  saved the last {t[-1] - t[0] + self._dt:.1f} s to {os.path.abspath(path)}")
        except Exception as exc:  # noqa: BLE001  (diagnostics must never block the stop)
            print(f"  (could not write stop details: {exc})")

    def _check_safety(self, targets: torch.Tensor, action: torch.Tensor) -> None:
        d = self.robot.data
        self._history.append(
            (
                self._step * self._dt,
                d.joint_pos.detach().cpu().numpy().copy(),
                d.joint_vel.detach().cpu().numpy().copy(),
                action.detach().cpu().numpy().copy(),
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


class SitHoldPolicy(SeatedSafety, Policy):
    """Hold the seated pose; stop on tilt, speed, torque or tracking limits."""

    task_name = "laai_sit_hold"

    def __init__(self, cfg, controller):
        torch.set_num_threads(1)
        super().__init__(cfg, controller)
        _require_pose(cfg.bike)
        self.robot = controller.robot
        self.device = cfg.device
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)  # the seated pose
        self.last_action = torch.zeros(
            self.robot.num_joints, dtype=torch.float32, device=self.device
        )
        self._init_safety(controller)

    # Interface used by robot/dry_run.py (same name as booster_deploy's LocomotionPolicy).
    def _action_to_targets(self, action: torch.Tensor) -> torch.Tensor:
        return self.default_joint_pos + action

    def reset(self) -> None:
        self._reset_safety()

    def inference(self) -> torch.Tensor:
        targets = self._handover(self.default_joint_pos.clone())
        self.last_action = targets - self.default_joint_pos
        self._check_safety(targets, self.last_action)
        return targets


class SitPolicy(SeatedSafety, LocomotionPolicy):
    """Run the trained sit policy around the seated pose, with the sit safety stops and the
    walk task's shake watchdog."""

    task_name = "laai_sit"

    def __init__(self, cfg, controller):
        # One CPU thread: faster on the robot (see laai_walk.py).
        torch.set_num_threads(1)
        _require_pose(cfg.bike)
        path = cfg.checkpoint_path
        if not os.path.isabs(path):
            path = os.path.join(_HERE, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No sit policy at {path}. Export one with experiments/sit/export.py, keep it as "
                "experiments/sit/runs/runN/policy.pt and install it with robot/install.sh sit."
            )
        super().__init__(cfg, controller)  # LocomotionPolicy: loads the model
        self._prev_action = None
        self._jump_count = 0
        self._jumps = deque(maxlen=SHAKE_WINDOW)
        self._init_safety(controller)

    def reset(self) -> None:
        super().reset()  # LocomotionPolicy: observation history and last action
        self._reset_safety()
        self._prev_action = None
        self._jump_count = 0
        self._jumps.clear()

    def _check_shaking(self, action: torch.Tensor) -> None:
        if self._prev_action is not None:
            jump = float((action - self._prev_action).norm())
            self._jump_count = self._jump_count + 1 if jump > MAX_ACTION_JUMP else 0
            if self._jump_count >= JUMP_STEPS:
                self._trip(f"action jumps > {MAX_ACTION_JUMP} for {JUMP_STEPS} steps (shaking)")
            self._jumps.append(jump)
            if len(self._jumps) == SHAKE_WINDOW:
                mean_jump = sum(self._jumps) / SHAKE_WINDOW
                if mean_jump > MAX_MEAN_JUMP:
                    self._trip(
                        f"average action jump {mean_jump:.2f} over "
                        f"{SHAKE_WINDOW * self._dt:.1f} s > {MAX_MEAN_JUMP} (shaking)"
                    )
        self._prev_action = action

    def inference(self) -> torch.Tensor:
        was_running = self.controller.is_running
        targets = self._handover(super().inference())  # seated pose + 0.25 x action
        action = self.last_action.clone()
        if was_running and not self.controller.is_running and not self._stopped:
            # booster_deploy's own fall check (in compute_observation) stopped the controller.
            self._trip("booster_deploy fall check (body tilted more than about 60 deg)")
        self._check_shaking(action)
        self._check_safety(targets, action)
        return targets


def _seated_robot(robot, bike: str):
    """The K1 with the seated pose as its default pose, and X sliding to it with the k1_walk
    gains (Booster's walking mode never runs)."""
    seated, _ = seated_targets(list(robot.joint_names), robot.default_joint_pos, bike)
    return robot.replace(
        default_joint_pos=seated,
        prepare_mode="standing",
        prepare_state=PrepareStateCfg(
            stiffness=list(robot.joint_stiffness),
            damping=list(robot.joint_damping),
            joint_pos=seated,
        ),
    )


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
        self.robot = _seated_robot(self.robot, self.bike)
        self.vel_command = None
        self.booster = BoosterRobotControllerCfg(exit_mode="damping")
        self.policy = LaaiSitHoldPolicyCfg(
            policy_joint_names=names,
            bike=self.bike,
            checkpoint_path=f"models/seated_poses/{self.bike}.json",
        )


@configclass
class LaaiSitPolicyCfg(K1LocomotionPolicyCfg):
    constructor = SitPolicy
    bike: str = BIKE


@configclass
class LaaiSitTaskCfg(K1WalkControllerCfg):
    bike: str = BIKE
    policy_file: str = POLICY_FILE  # relative to this file's folder, or absolute
    ankle_kd: float | None = None  # damping of the four ankle joints (None: k1_walk's 1.0)

    def __post_init__(self):
        super().__post_init__()
        joints = self.policy.policy_joint_names  # k1_walk's 20 joints, in its order
        if self.ankle_kd is not None:
            kd = list(self.robot.joint_damping)
            for name in ANKLE_JOINTS:
                kd[self.robot.joint_names.index(name)] = self.ankle_kd
            self.robot = self.robot.replace(joint_damping=kd)
        self.robot = _seated_robot(self.robot, self.bike)  # X uses the same gains
        self.vel_command = None  # the policy always sees a zero velocity command
        self.booster = BoosterRobotControllerCfg(exit_mode="damping")
        self.policy = LaaiSitPolicyCfg(
            policy_joint_names=joints,
            bike=self.bike,
            checkpoint_path=self.policy_file,
            # Same interface as the walk policy (laai_walk.py) and the sit training task.
            actor_obs_history_length=10,
            history_init="repeat",
            obs_dof_vel_scale=0.1,
            action_scale=0.25,
            use_actor_module=False,
            action_filter=1.0,
            arm_action_fix_joint_name=None,
            arm_action_fix_offset=0.0,
        )


register_task("laai_sit_hold", LaaiSitHoldTaskCfg())
register_task("laai_sit", LaaiSitTaskCfg())
# More ankle damping. Robot test 2026-11-19: after a push, the right ankle roll oscillated at
# about 9 Hz around an almost constant target (the joint swinging 3x further than commanded,
# with the feet barely touching the floor) until the speed stop, as laai_walk's ankles did with
# kd 1.0 (laai_walk.py). 1.5 is inside the sit training's damping randomization (x0.6 to x1.6).
register_task("laai_sit_ankle_kd15", LaaiSitTaskCfg(ankle_kd=ANKLE_KD))
