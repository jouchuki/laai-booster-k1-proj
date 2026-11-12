"""First RL training run on a Mac (CPU only).

Trains PPO on MuJoCo's Walker2d (a 2D two-legged walker) with Stable-Baselines3,
then shows the result in the MuJoCo viewer.

  pip install "gymnasium[mujoco]" stable-baselines3 tensorboard
  python train_walker.py train      # ~30-60 min on a laptop CPU
  tensorboard --logdir runs         # watch the reward go up (optional, 2nd terminal)
  python train_walker.py play       # watch the trained walker
"""

import sys

import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "Walker2d-v5"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000


def train():
    env = gym.make(ENV_ID)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs", device="cpu")
    model.learn(total_timesteps=STEPS)
    model.save("walker_ppo")
    print("Saved walker_ppo.zip")


def play():
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load("walker_ppo", device="cpu")
    obs, _ = env.reset()
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    {"train": train, "play": play}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
