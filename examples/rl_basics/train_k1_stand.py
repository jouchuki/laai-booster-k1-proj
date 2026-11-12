"""Teach the Booster K1 to stand and keep its balance when pushed (MuJoCo, CPU, runs on an M1 Mac).

Uses Booster's official K1 model from booster_assets, PD joint control like the real robot,
and PPO from Stable-Baselines3 with several simulations running in parallel.

Setup (once, in your conda env):
  pip install "gymnasium[mujoco]" stable-baselines3 tensorboard
  (booster_assets must already be installed: pip install -e ~/booster/booster_assets)

Commands:
  python train_k1_stand.py check            # 10 s sanity check, no training
  python train_k1_stand.py train            # train (default 5M steps)
  python train_k1_stand.py train 1000000    # train a shorter run
  python train_k1_stand.py eval             # score the trained policy, no window
  mjpython train_k1_stand.py play           # watch it; press P in the window to push the robot
  tensorboard --logdir runs_k1              # learning curves (second terminal)
"""

import os
import sys
import time
from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from booster_assets import BOOSTER_ASSETS_DIR

MODEL_PATH = os.path.join(BOOSTER_ASSETS_DIR, "robots/K1/K1_22dof.xml")
RUN_NAME = "k1_stand"

# Joint settings copied from Booster's own K1 config (booster_deploy/robots/k1.py).
# Order = the model's actuator order: head (2), left arm (4), right arm (4), left leg (6), right leg (6).
DEFAULT_POSE = np.array(
    [
        0,
        0,
        0.0,
        -1.3,
        0,
        0,
        0.0,
        1.3,
        0,
        0,
        0,
        0,
        0,
        0.105,
        -0.10,
        0,
        0,
        0,
        0,
        0.105,
        -0.10,
        0,
    ],
    dtype=np.float64,
)
KP = np.array([4.0] * 10 + [80, 80, 80, 80, 30, 30] * 2)
KD = np.array([1.0] * 10 + [2.0] * 12)
TORQUE_LIMIT = np.array([6, 6] + [14] * 8 + [30, 35, 20, 40, 20, 20] * 2, dtype=np.float64)

POLICY_HZ = 50  # the policy acts 50 times per second, like Booster's real controllers
PHYSICS_DT = 0.002  # physics step (s)
DECIMATION = round(1 / (POLICY_HZ * PHYSICS_DT))
ACTION_SCALE = 0.25  # action of +-1 moves a joint target +-0.25 rad from the default pose
EPISODE_SECONDS = 20
PUSH_EVERY = (2.0, 4.0)  # seconds between random pushes
PUSH_SPEED = 0.6  # max sideways/forward speed kick from a push (m/s)


class K1StandEnv(gym.Env):
    """Observation: body tilt, rotation speed, velocity, joint angles/speeds, last action.
    Action: joint target offsets for all 22 joints. Reward: stay upright, at height, near the default pose, calm."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, pushes=True, seed=None):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.pushes = pushes
        self.rng = np.random.default_rng(seed)
        n = self.model.nu
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9 + 3 * n,), np.float32)
        self.max_steps = EPISODE_SECONDS * POLICY_HZ
        self.stand_height = self._settle_height()

    # --- helpers -------------------------------------------------------------------------
    def _set_default(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.60
        self.data.qpos[7:] = DEFAULT_POSE

    def _settle_height(self):
        """Let the robot drop onto its feet once (0.3 s) to learn its normal standing height.
        Without a policy it would tip over after ~2 s: balancing is what PPO has to learn."""
        self._set_default()
        for _ in range(int(0.3 / PHYSICS_DT)):
            self._apply_pd(DEFAULT_POSE)
            mujoco.mj_step(self.model, self.data)
        return float(self.data.qpos[2])

    def _apply_pd(self, target):
        q, qd = self.data.qpos[7:], self.data.qvel[6:]
        torque = KP * (target - q) - KD * qd
        self.data.ctrl[:] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

    def _gravity_in_body(self):
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        return rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def _obs(self):
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        lin_vel_body = rot.reshape(3, 3).T @ self.data.qvel[0:3]
        return np.concatenate(
            [
                self._gravity_in_body(),  # which way is down (tilt)
                self.data.qvel[3:6] * 0.25,  # body rotation speed
                lin_vel_body,  # body velocity
                self.data.qpos[7:] - DEFAULT_POSE,  # joint angles
                self.data.qvel[6:] * 0.05,  # joint speeds
                self.last_action,
            ]
        ).astype(np.float32)

    # --- gym API -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._set_default()
        self.data.qpos[2] = self.stand_height
        self.data.qpos[7:] += self.rng.uniform(
            -0.05, 0.05, self.model.nu
        )  # small random start pose
        self.data.qvel[0:2] = self.rng.uniform(-0.1, 0.1, 2)
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(self.model.nu)
        self.steps = 0
        self.next_push = self.rng.uniform(*PUSH_EVERY)
        return self._obs(), {}

    def push(self, speed=PUSH_SPEED):
        self.data.qvel[0:2] += self.rng.uniform(-speed, speed, 2)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        target = DEFAULT_POSE + ACTION_SCALE * action
        for _ in range(DECIMATION):
            self._apply_pd(target)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        t = self.steps / POLICY_HZ
        if self.pushes and t >= self.next_push:
            self.push()
            self.next_push = t + self.rng.uniform(*PUSH_EVERY)

        grav = self._gravity_in_body()
        height = float(self.data.qpos[2])
        tilt = float(np.linalg.norm(grav[:2]))  # 0 = perfectly upright
        pose_err = float(np.sum((self.data.qpos[7:] - DEFAULT_POSE) ** 2))
        action_rate = float(np.sum((action - self.last_action) ** 2))
        joint_speed = float(np.sum(self.data.qvel[6:] ** 2))
        horiz_speed = float(np.sum(self.data.qvel[0:2] ** 2))

        reward = (
            1.0  # alive bonus
            + 1.0 * np.exp(-(tilt**2) / 0.02)  # stay upright
            + 1.0 * np.exp(-((height - self.stand_height) ** 2) / 0.002)  # stay at standing height
            + 0.5 * np.exp(-pose_err / 0.5)  # stay close to the default pose
            + 0.5 * np.exp(-horiz_speed / 0.1)  # don't drift around
            - 0.02 * action_rate  # smooth actions
            - 0.0005 * joint_speed  # calm joints
        )
        fell = height < 0.6 * self.stand_height or tilt > 0.7
        if fell:
            reward -= 10.0
        self.last_action = action
        truncated = self.steps >= self.max_steps
        return self._obs(), float(reward), bool(fell), bool(truncated), {"fell": fell}


# --- training / evaluation ---------------------------------------------------------------
def make_env(rank):
    def _init():
        return K1StandEnv(seed=rank)

    return _init


def train(total_steps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    n_envs = max(2, min(8, (os.cpu_count() or 2)))
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)]))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096 // n_envs,
        batch_size=1024,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.0},
        tensorboard_log="runs_k1",
        device="cpu",
        verbose=1,
    )
    print(f"Training with {n_envs} parallel simulations for {total_steps:,} steps...")
    start = time.time()
    model.learn(total_timesteps=total_steps, tb_log_name=RUN_NAME)
    model.save(RUN_NAME)
    env.save(f"{RUN_NAME}_vecnorm.pkl")
    print(
        f"Done in {(time.time() - start) / 60:.1f} min. Saved {RUN_NAME}.zip and {RUN_NAME}_vecnorm.pkl"
    )


def load_policy():
    import pickle

    from stable_baselines3 import PPO

    model = PPO.load(RUN_NAME, device="cpu")
    with open(f"{RUN_NAME}_vecnorm.pkl", "rb") as f:
        vecnorm = pickle.load(f)

    def act(obs):
        norm = np.clip(
            (obs - vecnorm.obs_rms.mean) / np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon), -10, 10
        )
        action, _ = model.predict(norm.astype(np.float32), deterministic=True)
        return action

    return act


def evaluate(episodes=10):
    act = load_policy()
    env = K1StandEnv(seed=123)
    falls, lengths = 0, []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, fell, trunc, _ = env.step(act(obs))
            done = fell or trunc
        falls += int(fell)
        lengths.append(env.steps / POLICY_HZ)
    print(
        f"{episodes - falls}/{episodes} episodes survived {EPISODE_SECONDS} s of random pushes "
        f"(average {np.mean(lengths):.1f} s standing)"
    )


def check():
    """No training: does the model load, and how long does it stand doing nothing?"""
    for pushes in (False, True):
        env = K1StandEnv(pushes=pushes, seed=0)
        env.reset()
        start = time.time()
        fell = trunc = False
        while not (fell or trunc):
            _, _, fell, trunc, _ = env.step(np.zeros(env.model.nu))
        rate = env.steps / (time.time() - start)
        print(
            f"pushes={'on ' if pushes else 'off'}: zero-action robot "
            f"{'FELL after' if fell else 'stood for'} {env.steps / POLICY_HZ:.1f} s "
            f"(standing height {env.stand_height:.2f} m, {rate:.0f} steps/s on one core)"
        )


def play():
    import mujoco.viewer

    act = load_policy()
    env = K1StandEnv(seed=7)
    obs, _ = env.reset()
    pending_push = []

    def on_key(keycode):
        if keycode in (ord("P"), ord("p")):
            pending_push.append(True)

    print("Press P in the viewer window to push the robot. Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            if pending_push:
                pending_push.clear()
                env.push(1.0)
            obs, _, fell, trunc, _ = env.step(act(obs))
            if fell or trunc:
                obs, _ = env.reset()
            viewer.cam.lookat[:] = env.data.qpos[0:3]
            viewer.sync()
            time.sleep(max(0.0, 1.0 / POLICY_HZ - (time.time() - t0)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000)
    elif cmd == "eval":
        evaluate()
    elif cmd == "play":
        play()
    else:
        check()
