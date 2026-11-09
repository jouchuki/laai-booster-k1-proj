"""booster_deploy task "k1_my_walk": runs YOUR walking policy trained with k1_task.py (mjlab),
with extra safety for first tests on the real robot.

Copy this file to:   booster_deploy/tasks/locomotion/robots/k1/k1_my_walk.py
Copy your policy to: booster_deploy/tasks/locomotion/robots/k1/models/k1_my_walk.pt

Simulation (Mac):  mjpython scripts/deploy.py --task k1_my_walk --mujoco
Real robot:        python3 scripts/deploy.py --task k1_my_walk

Robot settings (default pose, PD gains, effort limits, joint order) are Booster's own k1_walk
settings. The exported model already contains the observation normalization; no action filter.

Safety additions (on top of booster_deploy's own 60-degree fall stop):
  * Speed limits: even a full stick only asks for 0.5 m/s forward, 0.2 m/s back/sideways,
    0.6 rad/s turning (all well inside what the policy was trained on).
  * Shake watchdog: stops the policy (the robot falls back to Booster's own walking mode) if
      - any joint moves faster than MAX_JOINT_VEL rad/s, or
      - the policy's commands jump by more than MAX_ACTION_JUMP for JUMP_STEPS steps in a row
        (fast back-and-forth = shaking), or
      - the body tilts more than MAX_TILT_DEG.
    Thresholds were set from Booster's own walk policy in simulation, which never exceeds
    10.4 rad/s joint speed or 1.9 action jump, even at 1 m/s.
"""
import math

import torch

from booster_deploy.controllers.controller_cfg import VelocityCommandCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...locomotion import K1LocomotionPolicyCfg, LocomotionPolicy
from . import K1WalkControllerCfg

MAX_JOINT_VEL = 15.0     # rad/s
MAX_ACTION_JUMP = 2.5    # norm of the change in the 20 actions between two steps
JUMP_STEPS = 3           # that many steps in a row
MAX_TILT_DEG = 35.0      # body tilt from upright


class SafeLocomotionPolicy(LocomotionPolicy):
    def __init__(self, cfg, controller):
        super().__init__(cfg, controller)
        self._prev_action = None
        self._jump_count = 0
        self._stopped = False

    def reset(self) -> None:
        super().reset()
        self._prev_action = None
        self._jump_count = 0
        self._stopped = False

    def _trip(self, reason: str) -> None:
        if not self._stopped:
            self._stopped = True
            print(f"\n[SAFETY] {reason} -> stopping k1_my_walk (Booster's walking mode takes over).")
            self.controller.stop()

    def inference(self) -> torch.Tensor:
        targets = super().inference()
        action = self.last_action.clone()

        joint_vel = self.robot.data.joint_vel[self.real2sim_joint_map].abs().max().item()
        if joint_vel > MAX_JOINT_VEL:
            self._trip(f"joint speed {joint_vel:.1f} rad/s > {MAX_JOINT_VEL}")

        if self._prev_action is not None:
            jump = (action - self._prev_action).norm().item()
            self._jump_count = self._jump_count + 1 if jump > MAX_ACTION_JUMP else 0
            if self._jump_count >= JUMP_STEPS:
                self._trip(f"action jumps > {MAX_ACTION_JUMP} for {JUMP_STEPS} steps (shaking)")
        self._prev_action = action

        gravity = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w,
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gravity[2].item()))))
        if tilt > MAX_TILT_DEG:
            self._trip(f"body tilt {tilt:.0f} deg > {MAX_TILT_DEG}")
        return targets


@configclass
class K1MyWalkPolicyCfg(K1LocomotionPolicyCfg):
    constructor = SafeLocomotionPolicy


@configclass
class K1MyWalkTaskCfg(K1WalkControllerCfg):
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_forward_max=0.5,
        vx_backward_max=0.2,
        vy_max=0.2,
        vyaw_max=0.6,
    )

    def __post_init__(self):
        super().__post_init__()
        self.policy = K1MyWalkPolicyCfg(
            policy_joint_names=self.policy.policy_joint_names,
            # Relative to this file's folder (tasks/locomotion/robots/k1/).
            checkpoint_path="models/k1_my_walk.pt",
            actor_obs_history_length=10,
            history_init="repeat",
            obs_dof_vel_scale=0.1,
            action_scale=0.25,
            use_actor_module=False,
            action_filter=1.0,
            arm_action_fix_joint_name=None,
            arm_action_fix_offset=0.0,
        )


register_task("k1_my_walk", K1MyWalkTaskCfg())
