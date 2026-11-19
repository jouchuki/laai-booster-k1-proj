"""A shared, deterministic-action test after training, using separate simulated worlds.

`evaluate_policy` is the original test: pushes of up to 0.2 m/s at 2, 4, 6 and 8 s.
`push_sweep` repeats it at stronger pushes, for the policy and for a zero-action baseline (the
robot holding the seated pose). At 0.2 m/s the baseline already passes every episode, so a
policy has learned to balance only where it beats the baseline.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from tensordict import TensorDict

EVAL_PUSHES_MPS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)


def _make_env(device, num_envs, seed):
    # Import here to avoid a cycle during task registration.
    from experiments.sit_free.task import free_sit_env_cfg

    cfg = free_sit_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.observations["actor"].enable_corruption = False
    # Fixed perturbation times/velocities, independent of policy-dependent reset RNG.
    cfg.events.pop("push_robot")
    return ManagerBasedRlEnv(cfg, device=str(device))


@torch.inference_mode()
def _run(env, policy, device, num_envs, seed, push_mps):
    """One episode per world with scheduled pushes; `policy` None means zero actions."""
    rng = torch.Generator().manual_seed(seed)
    pushes = 2 * torch.rand(4, num_envs, 6, generator=rng) - 1
    pushes *= torch.tensor([push_mps, push_mps, 0.0, 0.3, 0.3, 0.3])
    pushes = pushes.to(device)
    obs, _ = env.reset(seed=seed)
    complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success = torch.zeros_like(complete)
    bike_falls = torch.zeros_like(complete)
    seat_falls = torch.zeros_like(complete)
    lengths = torch.zeros(num_envs, device=device)
    returns = torch.zeros_like(lengths)
    zero = torch.zeros(num_envs, env.action_manager.total_action_dim, device=device)
    for step in range(env.max_episode_length):
        if step in (100, 200, 300, 400):
            env.scene["robot"].write_root_link_velocity_to_sim(pushes[step // 100 - 1])
        live = ~complete
        actions = zero if policy is None else policy(TensorDict(obs, batch_size=[num_envs]))
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


def evaluate_policy(policy, device, *, num_envs=32, seed=12345, push_mps=0.2):
    env = _make_env(device, num_envs, seed)
    try:
        return _run(env, policy, device, num_envs, seed, push_mps)
    finally:
        env.close()


def push_sweep(policy, device, *, pushes=EVAL_PUSHES_MPS, num_envs=32, seed=12345):
    """Success fraction at each push speed, for `policy` and for zero actions."""
    env = _make_env(device, num_envs, seed)
    try:
        policy_success, zero_success = [], []
        for push in pushes:
            policy_success.append(
                _run(env, policy, device, num_envs, seed, push)["success_fraction"]
            )
            zero_success.append(_run(env, None, device, num_envs, seed, push)["success_fraction"])
    finally:
        env.close()
    return {
        "pushes_mps": list(pushes),
        "success_fraction": policy_success,
        "zero_action_success_fraction": zero_success,
        # Pushes where the policy keeps more episodes up than holding the pose does.
        "beats_zero_action_at_mps": [
            p for p, a, b in zip(pushes, policy_success, zero_success, strict=True) if a > b
        ],
    }
