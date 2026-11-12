"""Rehearse the robot task "laai_sit_hold" in simulation, on the bike, before a robot test.

booster_deploy's own MuJoCo mode loads the robot alone (no bike), so it cannot show this task.
This script runs the same task file (experiments/sit/deploy/laai_sit.py) through booster_deploy's
controller interface in our K1 + bike scene, with booster_deploy's control law (PD at 500 Hz,
policy at 50 Hz, effort limits), in the same order as on the robot:

  1. the robot starts near the seated pose, joints off by up to --offset rad (placed by hand);
  2. "X": booster_deploy's standing preparation slides all joints to the seated pose in 1 s,
     while a helper holds the bike upright;
  3. "A": the helper lets go and the task holds the pose for --seconds, safety stops active.

It prints the weight split (saddle, feet, hands), the tilt, the largest torque against its
limit and any safety stop. --push N pushes the trunk sideways with N newtons for 0.3 s halfway
through, to see the robot fall (it does not balance) and the tilt stop trip.

Linux and macOS, from the repository root (headless report):
    uv run --group sim python experiments/sit/check_hold.py
    uv run --group sim python experiments/sit/check_hold.py --offset 0.15 --push 60

macOS host, with the viewer:
    uv run --group sim mjpython experiments/sit/check_hold.py --viewer
Linux host, with the viewer:
    uv run --group sim python experiments/sit/check_hold.py --viewer
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = REPO_ROOT / "experiments" / "sit" / "deploy" / "laai_sit.py"
PREPARE_STEPS = 500  # booster_deploy's standing preparation: 500 commands, 2 ms apart (1 s)


def booster_deploy_dir() -> Path | None:
    for path in (REPO_ROOT / "booster_deploy", REPO_ROOT / "booster" / "booster_deploy"):
        if (path / "booster_deploy").is_dir():
            return path
    return None


def load_task_module():
    """Import the repository's laai_sit.py as booster_deploy's task module, without installing it."""
    root = booster_deploy_dir()
    if root is None:
        raise ModuleNotFoundError("booster_deploy not found; run: bash scripts/setup_booster.sh")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    name = "tasks.locomotion.robots.k1.laai_sit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TASK_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_controller(task, bike: str):
    """A booster_deploy controller that drives the K1 in our K1 + bike scene."""
    import torch
    from booster_deploy.controllers.base_controller import BaseController

    from k1bike.paths import BIKES
    from k1bike.scene import k1_seated_bike_scene

    class SeatedSceneController(BaseController):
        def __init__(self, cfg):
            super().__init__(cfg)
            spec = k1_seated_bike_scene(bike_xml=BIKES[bike])
            # A helper holding the bike upright (welded to the world), as a person does during
            # X; released when the task starts.
            spec.add_equality(
                name="helper_holds_bike",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1="bike/frame",
            )
            self.model = spec.compile()
            self.model.opt.timestep = cfg.mujoco.physics_dt  # 2 ms, as in booster_deploy's MuJoCo
            self.data = mujoco.MjData(self.model)
            names = list(cfg.robot.joint_names)
            joints = [self.model.joint(n).id for n in names]
            self.qadr = np.array([self.model.jnt_qposadr[j] for j in joints])
            self.dadr = np.array([self.model.jnt_dofadr[j] for j in joints])
            by_joint = {self.model.actuator_trnid[a, 0]: a for a in range(self.model.nu)}
            self.act = np.array([by_joint[j] for j in joints])
            self.root = self.model.jnt_qposadr[self.model.joint("world_joint").id]
            self.root_dof = self.model.jnt_dofadr[self.model.joint("world_joint").id]
            self.kp = self.robot.joint_stiffness.numpy()
            self.kd = self.robot.joint_damping.numpy()
            self.limit = self.robot.effort_limit.numpy()
            self.push = np.zeros(3)
            self.torque_ratio = 0.0

        def update_state(self):
            d, r, rd = self.data, self.root, self.root_dof
            data = self.robot.data
            data.joint_pos = torch.from_numpy(d.qpos[self.qadr].astype(np.float32))
            data.joint_vel = torch.from_numpy(d.qvel[self.dadr].astype(np.float32))
            data.root_pos_w = torch.from_numpy(d.qpos[r : r + 3].astype(np.float32))
            data.root_quat_w = torch.from_numpy(d.qpos[r + 3 : r + 7].astype(np.float32))
            data.root_lin_vel_b = torch.from_numpy(d.qvel[rd : rd + 3].astype(np.float32))
            data.root_ang_vel_b = torch.from_numpy(d.qvel[rd + 3 : rd + 6].astype(np.float32))

        def pd_steps(self, targets: np.ndarray, steps: int) -> None:
            trunk = self.model.body("trunk").id
            for _ in range(steps):
                q, dq = self.data.qpos[self.qadr], self.data.qvel[self.dadr]
                tau = np.clip(self.kp * (targets - q) - self.kd * dq, -self.limit, self.limit)
                self.torque_ratio = max(self.torque_ratio, float(np.max(np.abs(tau) / self.limit)))
                self.data.ctrl[self.act] = tau
                self.data.xfrc_applied[trunk, :3] = self.push
                mujoco.mj_step(self.model, self.data)

        def ctrl_step(self, dof_targets):
            self.pd_steps(dof_targets.cpu().numpy(), self.cfg.mujoco.decimation)

        def run(self):
            raise NotImplementedError("use rehearse()")

    from copy import deepcopy

    return SeatedSceneController(deepcopy(task.LaaiSitHoldTaskCfg(bike=bike)))


def load_split(model, data) -> dict:
    """Vertical load on the saddle and the feet (N); the hands carry the rest of the weight."""
    from k1bike.saddle import saddle_geom_ids

    saddle = set(saddle_geom_ids(model, "bike/"))
    floor = model.geom("floor").id
    force, out = np.zeros(6), {"saddle": 0.0, "feet": 0.0}
    for i, c in enumerate(data.contact[: data.ncon]):
        bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
        if all(b == 0 or model.body(b).name.startswith("bike/") for b in bodies):
            continue
        mujoco.mj_contactForce(model, data, i, force)
        if floor in (c.geom1, c.geom2):
            out["feet"] += force[0] * abs(c.frame[2])
        elif saddle & {c.geom1, c.geom2}:
            out["saddle"] += force[0] * abs(c.frame[2])
    return out


def rehearse(
    bike: str = "puky_bike",
    seconds: float = 6.0,
    offset: float = 0.1,
    push: float = 0.0,
    seed: int = 0,
    viewer: bool = False,
) -> dict:
    """Run the robot sequence (placed by hand, X, A, hold) and return what happened."""
    task = load_task_module()
    ctl = make_controller(task, bike)
    if not viewer:
        return _rehearse(ctl, bike, seconds, offset, push, seed, None)
    import mujoco.viewer

    with mujoco.viewer.launch_passive(ctl.model, ctl.data) as window:
        return _rehearse(ctl, bike, seconds, offset, push, seed, window)


def _rehearse(ctl, bike, seconds, offset, push, seed, viewer) -> dict:
    m, d = ctl.model, ctl.data
    seated = np.array(ctl.cfg.robot.default_joint_pos)

    # 1. Placed by hand: the seated keyframe with every joint off by up to `offset` rad.
    mujoco.mj_resetDataKeyframe(m, d, m.key("seated").id)
    rng = np.random.default_rng(seed)
    d.qpos[ctl.qadr] = seated + rng.uniform(-offset, offset, size=seated.size)
    mujoco.mj_forward(m, d)
    # 2. X: slide from the current joint angles to the seated pose (prepare gains = hold gains).
    start = d.qpos[ctl.qadr].copy()
    for i in range(PREPARE_STEPS):
        ctl.pd_steps(start + (i + 1) / PREPARE_STEPS * (seated - start), 1)
        if viewer is not None and i % 10 == 0:
            viewer.sync()
    # 3. A: the task holds the pose; the helper lets go of the bike.
    d.eq_active[m.equality("helper_holds_bike").id] = 0
    ctl.update_state()
    ctl.start()
    weight = m.body_subtreemass[m.body("trunk").id] * 9.81
    loads, tilts = [], []
    steps = int(seconds / ctl.cfg.policy_dt)
    for k in range(steps):
        ctl.push[:] = [0, push, 0] if push and steps // 2 <= k < steps // 2 + 15 else 0
        ctl.update_state()
        targets = ctl.policy_step()
        ctl.ctrl_step(targets)
        tilts.append(ctl.policy.tilt_change_deg())
        if k * ctl.cfg.policy_dt >= 1.0:
            loads.append(load_split(m, d))
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(ctl.cfg.policy_dt)
        if not ctl.is_running:
            break
    split = {
        key: float(np.mean([x[key] for x in loads])) / weight if loads else float("nan")
        for key in ("saddle", "feet")
    }
    split["hands"] = 1.0 - split["saddle"] - split["feet"]
    return {
        "bike": bike,
        "stopped": ctl.policy.stop_reason,
        "held_s": len(tilts) * ctl.cfg.policy_dt,
        "split": split,
        "max_tilt_change_deg": float(np.max(tilts)) if tilts else float("nan"),
        "max_torque_ratio": ctl.torque_ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bike", default="puky_bike")
    ap.add_argument("--seconds", type=float, default=6.0, help="how long the task holds the pose")
    ap.add_argument("--offset", type=float, default=0.1, help="start posture error per joint, rad")
    ap.add_argument("--push", type=float, default=0.0, help="sideways push on the trunk, N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    result = rehearse(
        bike=args.bike,
        seconds=args.seconds,
        offset=args.offset,
        push=args.push,
        seed=args.seed,
        viewer=args.viewer,
    )
    s = result["split"]
    print(f"bike {result['bike']}: held {result['held_s']:.1f} s")
    print(f"  weight: saddle {s['saddle']:.0%}, feet {s['feet']:.0%}, hands {s['hands']:.0%}")
    print(
        f"  largest trunk tilt from the seated orientation: {result['max_tilt_change_deg']:.1f} deg"
    )
    print(f"  largest joint torque: {result['max_torque_ratio']:.0%} of its limit")
    print(f"  safety stop: {result['stopped'] or 'none'}")


if __name__ == "__main__":
    main()
