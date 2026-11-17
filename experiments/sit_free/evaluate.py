"""A shared, deterministic-action test after training, using separate simulated worlds."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from tensordict import TensorDict


def evaluate_policy(policy, device, *, num_envs=32, seed=12345):
    # Import here to avoid a cycle during task registration.
    from experiments.sit_free.task import free_sit_env_cfg

    cfg = free_sit_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.observations["actor"].enable_corruption = False
    # Fixed perturbation times/velocities, independent of policy-dependent reset RNG.
    cfg.events.pop("push_robot")
    rng = torch.Generator().manual_seed(seed)
    pushes = 2 * torch.rand(4, num_envs, 6, generator=rng) - 1
    pushes *= torch.tensor([0.2, 0.2, 0.0, 0.3, 0.3, 0.3])
    pushes = pushes.to(device)
    env = ManagerBasedRlEnv(cfg, device=str(device))
    try:
        obs, _ = env.reset(seed=seed)
        complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
        success = torch.zeros_like(complete)
        bike_falls = torch.zeros_like(complete)
        seat_falls = torch.zeros_like(complete)
        lengths = torch.zeros(num_envs, device=device)
        returns = torch.zeros_like(lengths)
        with torch.inference_mode():
            for step in range(env.max_episode_length):
                if step in (100, 200, 300, 400):
                    env.scene["robot"].write_root_link_velocity_to_sim(pushes[step // 100 - 1])
                live = ~complete
                actions = policy(TensorDict(obs, batch_size=[num_envs]))
                obs, reward, terminated, truncated, _ = env.step(actions)
                lengths += live
                returns += reward * live
                finished = live & (terminated | truncated)
                success |= finished & truncated & ~terminated
                bike_falls |= finished & env.termination_manager.get_term("bike_fell")
                seat_falls |= finished & env.termination_manager.get_term("fell_off_seat")
                complete |= finished
                if complete.all():
                    break
        return {
            "seed": seed,
            "episodes": num_envs,
            "completed_episodes": int(complete.sum()),
            "success_fraction": float(success.float().mean()),
            "bike_fall_fraction": float(bike_falls.float().mean()),
            "seat_fall_fraction": float(seat_falls.float().mean()),
            "mean_episode_seconds": float(lengths.mean()) * env.step_dt,
            "mean_return": float(returns.mean()),
        }
    finally:
        env.close()
