"""Dry run: run a task's policy on live robot sensor data WITHOUT controlling the motors.

The script only subscribes to /low_state. It creates no /joint_ctrl publisher and makes no
mode-change (RPC) calls, so the robot stays under whatever controller is running (for example
Booster's own walking mode, standing still). Every 20 ms it builds the observation exactly like
the real deployment, runs the policy, computes the joint targets it WOULD send, and logs it all.
At the end it prints a report: sensor rates, IMU orientation, joint offsets, sensor noise, policy
outputs, would-be torques and whether the safety watchdog would have stopped the policy.

Note: the policy runs open-loop. The robot does not follow its targets, so after the first
second its own "last action" input no longer matches the robot's motion. The report's
"simulation reference" values come from the same open-loop setup in MuJoCo.

On the robot (robot standing in Booster's walking mode, harness recommended):
  cd ~/booster_deploy && source .venv/bin/activate
  source /opt/booster/BoosterRos2Interface/install/setup.bash
  python3 scripts/dry_run.py --task laai_walk --seconds 20

Rehearsal on the Mac (MuJoCo plays the robot; Booster's k1_walk keeps it standing):
  python scripts/dry_run.py --task laai_walk --source mujoco

Options: --vx/--vy/--vyaw set a constant velocity command (default 0), Ctrl+C ends early.
The full log is saved to logs/dry_run_<task>_<time>.npz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pkgutil
import sys
import threading
import time
from collections import deque

sys.path.append(".")

import numpy as np
import torch
from booster_deploy.controllers.base_controller import BaseController
from booster_deploy.utils.isaaclab import math as lab_math


# ----------------------------------------------------------------------------------------------
# State sources
# ----------------------------------------------------------------------------------------------
class RosLowStateSource:
    """Read-only /low_state subscriber (same topic, QoS and parsing as booster_robot_controller)."""

    realtime = True

    def __init__(self) -> None:
        import rclpy
        from booster_interface.msg import LowState
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("booster_dry_run_low_state_sub")
        self._node.create_subscription(
            LowState,
            "/low_state",
            self._callback,
            QoSProfile(
                depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST
            ),
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._lock = threading.Lock()
        self._latest = None
        self.count = 0
        self.motor_counts: set[int] = set()
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        while self._running and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _callback(self, msg) -> None:
        motors = msg.motor_state_serial
        state = {
            "t": time.perf_counter(),
            "rpy": np.array(msg.imu_state.rpy, dtype=np.float32),
            "gyro": np.array(msg.imu_state.gyro, dtype=np.float32),
            "q": np.array([m.q for m in motors], dtype=np.float32),
            "dq": np.array([m.dq for m in motors], dtype=np.float32),
            "tau": np.array([m.tau_est for m in motors], dtype=np.float32),
        }
        with self._lock:
            self._latest = state
            self.count += 1
            self.motor_counts.add(len(motors))

    def step(self) -> None:
        pass

    def latest(self):
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._executor.shutdown()
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


class MujocoSource:
    """MuJoCo stand-in for the robot: Booster's k1_walk keeps the robot standing, and the state is
    reported in the same form as /low_state (rpy, body-frame gyro, joint q/dq/tau)."""

    realtime = False

    def __init__(self, noise: bool) -> None:
        from copy import deepcopy

        from booster_deploy.controllers.mujoco_controller import MujocoController
        from booster_deploy.utils.registry import get_task

        class _Holder(MujocoController):
            def update_vel_command(self):  # zero command, no stdin reading
                pass

        self._sim = _Holder(deepcopy(get_task("k1_walk")))
        self._sim.update_state()
        self._sim.start()
        self._noise = noise
        self._rng = np.random.default_rng(0)
        self.count = 0
        self.motor_counts = {self._sim.robot.num_joints}
        for _ in range(100):  # 2 s to settle on its feet before the dry run starts
            self.step()

    def step(self) -> None:
        sim = self._sim
        sim.update_state()
        sim.ctrl_step(sim.policy_step())
        self.count += 1

    def latest(self):
        d = self._sim.mj_data
        quat = torch.tensor(d.qpos[3:7], dtype=torch.float32).unsqueeze(0)
        rpy = np.array([float(a) for a in lab_math.euler_xyz_from_quat(quat)], dtype=np.float32)
        state = {
            "t": time.perf_counter(),
            "rpy": rpy,
            "gyro": d.qvel[3:6].astype(np.float32).copy(),
            "q": d.qpos[7:].astype(np.float32).copy(),
            "dq": d.qvel[6:].astype(np.float32).copy(),
            "tau": d.qfrc_actuator[6:].astype(np.float32).copy(),
        }
        if self._noise:  # rough real-sensor noise
            state["dq"] += self._rng.normal(0, 0.5, state["dq"].shape).astype(np.float32)
            state["gyro"] += self._rng.normal(0, 0.05, 3).astype(np.float32)
        return state

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------------------------
# Dry-run controller: the real policy, fed like the robot, but ctrl_step sends nothing
# ----------------------------------------------------------------------------------------------
class DryRunController(BaseController):
    def __init__(self, cfg, source, command) -> None:
        cfg.policy.enable_safety_fallback = False  # tilt is logged instead; avoids per-step spam
        super().__init__(cfg)
        self.source = source
        self.command = command
        self.trips: list[tuple[int, str]] = []
        self.state = None

    def update_state(self) -> None:
        s = self.source.latest()
        self.state = s
        n = self.robot.num_joints
        data = self.robot.data
        data.joint_pos = torch.from_numpy(s["q"][:n].copy())
        data.joint_vel = torch.from_numpy(s["dq"][:n].copy())
        data.feedback_torque = torch.from_numpy(s["tau"][:n].copy())
        data.root_pos_w = torch.zeros(3)
        rpy = torch.from_numpy(s["rpy"].copy())
        data.root_quat_w = lab_math.quat_from_euler_xyz(*rpy).squeeze()
        data.root_lin_vel_b = torch.zeros(3)
        data.root_ang_vel_b = torch.from_numpy(s["gyro"].copy())

    def update_vel_command(self) -> None:
        if self.vel_command is None:
            return
        vx, vy, vyaw = self.command
        c = self.vel_command
        c.lin_vel_x = c.clamp_vx(vx)
        c.lin_vel_y = max(-c.vy_max, min(c.vy_max, vy))
        c.ang_vel_yaw = max(-c.vyaw_max, min(c.vyaw_max, vyaw))

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        pass  # DRY RUN: nothing is sent to the motors

    def stop(self) -> None:
        # The watchdog asks to stop: record it and keep logging instead.
        self.trips.append((self._step_count, "watchdog/safety stop requested"))

    def run(self) -> None:
        raise NotImplementedError("use dry_run()")


# ----------------------------------------------------------------------------------------------
def load_tasks() -> None:
    import tasks as tasks_pkg

    for mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod.name)


def watchdog_limits(cfg) -> dict:
    mod = sys.modules.get(getattr(cfg.policy.constructor, "__module__", ""), None)
    get = lambda name, default: float(getattr(mod, name, default))
    return {
        "joint_vel": get("MAX_JOINT_VEL", 15.0),
        "jump": get("MAX_ACTION_JUMP", 2.5),
        "jump_steps": int(get("JUMP_STEPS", 3)),
        "window": int(get("SHAKE_WINDOW", 10)),
        "mean_jump": get("MAX_MEAN_JUMP", 0.8),
        "tilt": get("MAX_TILT_DEG", 35.0),
    }


def policy_targets(policy, action: torch.Tensor) -> np.ndarray:
    """Targets the policy asks for, without the start-up handover blend."""
    if hasattr(policy, "_action_to_targets"):
        return policy._action_to_targets(action).detach().cpu().numpy().copy()
    return np.full(policy.robot.num_joints, np.nan, dtype=np.float32)


def dry_run(args) -> int:
    load_tasks()
    from copy import deepcopy

    from booster_deploy.utils.registry import get_task

    cfg = deepcopy(get_task(args.task))
    cfg.policy.device = "cpu"

    if args.source == "ros":
        source = RosLowStateSource()
    else:
        source = MujocoSource(noise=args.noise)

    print(f"DRY RUN of task '{args.task}': reading sensors only, NO motor commands are sent.")
    deadline = time.perf_counter() + 5.0
    while source.latest() is None:
        if time.perf_counter() > deadline:
            print(
                "No /low_state message within 5 s. Is the robot on, and did you source "
                "/opt/booster/BoosterRos2Interface/install/setup.bash?"
            )
            source.close()
            return 1
        time.sleep(0.05)

    ctl = DryRunController(cfg, source, (args.vx, args.vy, args.vyaw))
    policy = ctl.policy
    names = list(cfg.robot.joint_names)
    n = ctl.robot.num_joints
    limits = watchdog_limits(cfg)
    first = source.latest()
    if len(first["q"]) != n:
        print(f"WARNING: /low_state has {len(first['q'])} motors, the task expects {n}.")

    log = {
        k: []
        for k in (
            "t",
            "period",
            "age",
            "infer_ms",
            "rpy",
            "gyro",
            "gravity",
            "q",
            "dq",
            "action",
            "target",
            "policy_target",
        )
    }
    jumps: deque = deque(maxlen=limits["window"])
    prev_action = None
    ctl.update_state()
    ctl.update_vel_command()
    ctl.start()
    steps = int(args.seconds / cfg.policy_dt)
    dt = cfg.policy_dt
    t_start = next_t = time.perf_counter()
    last_print = 0.0
    last_loop = None
    print(f"Logging {args.seconds:.0f} s at {1 / dt:.0f} Hz. Ctrl+C to stop early.\n")
    try:
        for k in range(steps):
            if source.realtime:
                while time.perf_counter() < next_t:
                    time.sleep(0.0002)
                next_t += dt
            else:
                source.step()
            now = time.perf_counter()
            ctl.update_state()
            ctl.update_vel_command()
            t0 = time.perf_counter()
            target = ctl.policy_step()
            infer_ms = (time.perf_counter() - t0) * 1000
            ctl.ctrl_step(target)  # no-op

            s = ctl.state
            action = policy.last_action.detach().cpu().clone()
            gravity = lab_math.quat_apply_inverse(
                ctl.robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0])
            ).numpy()
            log["t"].append(now - t_start if source.realtime else k * dt)
            log["period"].append(np.nan if last_loop is None else now - last_loop)
            log["age"].append(now - s["t"] if source.realtime else 0.0)
            log["infer_ms"].append(infer_ms)
            log["rpy"].append(s["rpy"])
            log["gyro"].append(s["gyro"])
            log["gravity"].append(gravity)
            log["q"].append(s["q"][:n])
            log["dq"].append(s["dq"][:n])
            log["action"].append(action.numpy())
            log["target"].append(target.detach().cpu().numpy().copy())
            log["policy_target"].append(policy_targets(policy, action))
            last_loop = now
            if prev_action is not None:
                jumps.append(float((action - prev_action).norm()))
            prev_action = action

            if (log["t"][-1] - last_print >= 1.0) or k == steps - 1:
                last_print = log["t"][-1]
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(gravity[2])))))
                mj = sum(jumps) / len(jumps) if jumps else 0.0
                print(
                    f"t={log['t'][-1]:5.1f}s  tilt={tilt:4.1f}deg  max|joint vel|="
                    f"{np.abs(s['dq'][:n]).max():5.2f}  max|action|={action.abs().max():4.2f}  "
                    f"avg jump={mj:4.2f}  watchdog stops={len(ctl.trips)}"
                )
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        source.close()

    L = {k: np.array(v) for k, v in log.items() if len(v)}
    if "t" not in L or len(L["t"]) < 10:
        print("Too few steps logged for a report.")
        return 1
    os.makedirs("logs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("logs", f"dry_run_{args.task}_{args.source}_{stamp}.npz")
    meta = {
        "task": args.task,
        "source": args.source,
        "command": [args.vx, args.vy, args.vyaw],
        "joint_names": names,
        "policy_joint_names": list(cfg.policy.policy_joint_names),
        "checkpoint": cfg.policy.checkpoint_path,
        "policy_dt": dt,
        "default_joint_pos": list(map(float, cfg.robot.default_joint_pos)),
        "stiffness": list(map(float, cfg.robot.joint_stiffness)),
        "effort_limit": list(map(float, cfg.robot.effort_limit)),
        "low_state_messages": source.count,
        "motor_counts": sorted(source.motor_counts),
        "watchdog_trips": ctl.trips,
        "limits": limits,
    }
    np.savez(path, meta=json.dumps(meta), **L)
    report(L, meta, cfg, source, ctl, limits)
    print(f"\nFull log: {path}")
    return 0


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def report(L, meta, cfg, source, ctl, limits) -> None:
    names = meta["joint_names"]
    n = len(names)
    dt = meta["policy_dt"]
    duration = L["t"][-1] + dt
    default = np.array(meta["default_joint_pos"], dtype=np.float32)
    kp = np.array(meta["stiffness"], dtype=np.float32)
    effort = np.array(meta["effort_limit"], dtype=np.float32)
    status_counts = {"OK": 0, "WARN": 0, "FAIL": 0}

    def line(status, text, ref=""):
        status_counts[status] += 1
        print(
            f"  [{status:4s}] {text}" + (f"\n           simulation reference: {ref}" if ref else "")
        )

    print("\n================ DRY RUN REPORT ================")
    print(
        f"Task {meta['task']}, {len(L['t'])} policy steps over {duration:.1f} s, "
        f"command {tuple(meta['command'])}, source {meta['source']}\n"
    )

    print("Communication and timing")
    if source.realtime:
        rate = meta["low_state_messages"] / duration
        line("OK" if rate >= 100 else "WARN", f"/low_state rate {rate:.0f} Hz")
        motors = meta["motor_counts"]
        line("OK" if motors == [n] else "FAIL", f"motors per message {motors} (task expects {n})")
        per = L["period"][1:] * 1000
        late = np.mean(per > dt * 1000 + 5) * 100
        line(
            "OK" if per.mean() < dt * 1000 + 1 and per.max() < 2.5 * dt * 1000 else "WARN",
            f"policy loop {per.mean():.1f} ms average (target {dt * 1000:.0f}), max {per.max():.1f} ms, "
            f"{late:.1f}% of steps more than 5 ms late",
        )
        age = L["age"] * 1000
        line(
            "OK" if np.median(age) < 10 else "WARN",
            f"sensor data age at each step: median {np.median(age):.1f} ms, max {age.max():.1f} ms",
        )
    inf = L["infer_ms"]
    line(
        "OK" if np.percentile(inf, 99) < 10 else "WARN",
        f"policy inference {np.median(inf):.2f} ms median, {np.percentile(inf, 99):.2f} ms 99th percentile",
    )

    print("\nIMU")
    g = L["gravity"]
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
    gm = g.mean(0)
    line(
        "OK" if gm[2] < -0.95 else ("FAIL" if gm[2] > 0 else "WARN"),
        f"gravity in body frame {np.round(gm.astype(float), 3).tolist()} (upright robot: about [0, 0, -1]); "
        f"tilt {tilt.mean():.1f} deg average, {tilt.max():.1f} max",
    )
    gyro = L["gyro"]
    line(
        "OK" if np.abs(gyro.mean(0)).max() < 0.1 else "WARN",
        f"gyro mean {np.round(gyro.mean(0).astype(float), 3).tolist()} rad/s, noise std {np.round(gyro.std(0).astype(float), 3).tolist()}",
        "mean about 0 when the robot stands still",
    )

    print("\nJoints (robot order)")
    dev = (L["q"] - default[None]).mean(0)
    worst = np.argsort(-np.abs(dev))[:5]
    legs = [i for i, nm in enumerate(names) if any(p in nm for p in ("hip", "knee", "ankle"))]
    big = [i for i in range(n) if abs(dev[i]) > (0.3 if i in legs else 0.7)]
    line(
        "OK" if not big else "WARN",
        "largest offsets from the policy's default pose: "
        + ", ".join(f"{names[i]} {dev[i]:+.2f}" for i in worst),
        REFERENCE["offsets"],
    )
    dq_std = L["dq"].std(0)
    j = int(np.argmax(dq_std))
    line(
        "OK" if dq_std.max() < 1.5 else "WARN",
        f"joint velocity noise: max std {dq_std.max():.2f} rad/s at {names[j]}, "
        f"median {np.median(dq_std):.2f}",
        REFERENCE["dq_noise"],
    )
    jv = np.abs(L["dq"]).max()
    line(
        "OK" if jv < limits["joint_vel"] else "WARN",
        f"peak joint speed {jv:.2f} rad/s (watchdog limit {limits['joint_vel']:.0f})",
    )

    print("\nPolicy outputs (open loop: the robot does not follow them)")
    a = L["action"]
    first_s = a[: int(1 / dt)]
    line(
        "OK" if np.abs(first_s).max() < 1.6 else "WARN",
        f"first second: max |action| {np.abs(first_s).max():.2f}, mean {np.abs(first_s).mean():.2f}",
        REFERENCE.get("first_action", "n/a"),
    )
    line(
        "OK" if np.abs(a).max() < 1.6 else "WARN",
        f"whole run: max |action| {np.abs(a).max():.2f}, mean {np.abs(a).mean():.2f}",
        REFERENCE.get("all_action", "n/a"),
    )
    jumps = np.linalg.norm(np.diff(a, axis=0), axis=1)
    w = limits["window"]
    roll = np.convolve(jumps, np.ones(w) / w, "valid") if len(jumps) >= w else np.array([0.0])
    line(
        "OK" if roll.max() < limits["mean_jump"] * 0.75 else "WARN",
        f"action jump per step: median {np.median(jumps):.2f}, max {jumps.max():.2f}; "
        f"{w * dt:.1f} s average max {roll.max():.2f} (watchdog limit {limits['mean_jump']})",
        REFERENCE.get("jump", "n/a"),
    )
    pt = L["policy_target"]
    if np.isfinite(pt).all():
        torque = np.abs(kp[None] * (pt[: int(1 / dt)] - L["q"][: int(1 / dt)]))
        ratio = (torque / effort[None]).max(0)
        k = int(np.argmax(ratio))
        line(
            "OK" if ratio.max() < 1.0 else "WARN",
            f"would-be torque in the first second (kp x (target - position)): highest {torque.max(0)[k]:.1f} Nm "
            f"at {names[k]}, {ratio.max() * 100:.0f}% of its training effort limit",
            REFERENCE["torque"],
        )

    print("\nWatchdog")
    trips = [t for t in meta["watchdog_trips"]]
    tilt_over = np.mean(tilt > limits["tilt"]) * 100
    line(
        "OK" if not trips else "FAIL",
        "no watchdog stop requested"
        if not trips
        else f"watchdog would have stopped the policy {len(trips)} time(s), first at step {trips[0][0]} "
        f"({trips[0][0] * dt:.1f} s); see the [SAFETY] line above",
    )
    if tilt_over:
        line("WARN", f"tilt above {limits['tilt']:.0f} deg in {tilt_over:.0f}% of steps")

    print(
        f"\nResult: {status_counts['OK']} OK, {status_counts['WARN']} WARN, {status_counts['FAIL']} FAIL"
    )
    if status_counts["FAIL"]:
        print("Do NOT run the policy on the motors until the FAIL items are understood.")
    elif status_counts["WARN"]:
        print("Look at the WARN items before running the policy on the motors.")
    else:
        print("All checks passed. Next step: the harness test with deploy.py.")


# Values from the MuJoCo rehearsal (python scripts/dry_run.py --task laai_walk --source mujoco
# --noise): Booster's k1_walk holds the robot, laai_walk (walk run 2 policy) runs open loop.
REFERENCE: dict[str, str] = {
    "offsets": "elbow pitch +0.45/+0.37, elbow yaw -0.24/+0.23, knees about -0.18 (Booster's policy "
    "holds the arms differently). WARN above 0.3 rad on a leg or 0.7 rad on an arm: "
    "wrong joint order or sign, or the robot is not standing",
    "dq_noise": "0.0 without noise, 0.5 with --noise; training added up to +-1.5 rad/s",
    "first_action": "max 1.04-1.12, mean 0.20-0.21 (WARN above 1.6)",
    "all_action": "max 1.04-1.12, mean 0.20 (WARN above 1.6)",
    "jump": "median 0.00-0.10, 0.2 s average max 0.13-0.18",
    "torque": "11 Nm at left elbow pitch (about 80%): the arm offset of Booster's policy; "
    "the 0.5 s handover blend spreads this out on the robot",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", default="laai_walk")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vyaw", type=float, default=0.0)
    ap.add_argument(
        "--source",
        choices=("ros", "mujoco"),
        default="ros",
        help="ros = live robot (default); mujoco = rehearsal on the Mac",
    )
    ap.add_argument("--noise", action="store_true", help="mujoco only: add rough sensor noise")
    return dry_run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
