# RL basics (practice)

Practice scripts from learning PPO on a Mac with Stable-Baselines3, before the mjlab work.
They are not part of the training pipeline and need their own packages:

### Linux and macOS

```bash
uv run --with "gymnasium[mujoco]" --with stable-baselines3 --with tensorboard python examples/rl_basics/train_walker.py train
uv run --with "gymnasium[mujoco]" --with stable-baselines3 --with tensorboard python examples/rl_basics/train_k1_stand.py check
```

| Script | What |
|---|---|
| `train_walker.py` | PPO on Gymnasium's Walker2d |
| `train_k1_stand.py` | K1 learns to stand and resist pushes in MuJoCo (PD control, PPO) |

Outputs (models, TensorBoard runs) are written next to where you run them and are gitignored.
