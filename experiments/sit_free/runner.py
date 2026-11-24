"""Initialize a new experiment from an actor, without resuming its old training state."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg


@dataclass
class PolicyInitRunnerCfg(RslRlOnPolicyRunnerCfg):
    init_policy_from: str | None = None
    """Actor checkpoint for a new run; critic, optimizer and counters start fresh."""


def initialize_policy(runner, checkpoint: Path) -> dict:
    """Load the complete actor (including normalization and exploration distribution)."""
    checkpoint = checkpoint.resolve(strict=True)
    loaded = torch.load(checkpoint, map_location=runner.device, weights_only=True)
    # Use the algorithm directly: mjlab's runner.load also restores environment counters,
    # even when only the actor was requested. No old optimizer, critic or counters here.
    runner.alg.load(loaded, load_cfg={"actor": True}, strict=True)
    return {
        "mode": "pretrained_actor",
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_iteration": int(loaded["iter"]),
        "loaded": ["actor", "actor_observation_normalization", "actor_distribution"],
        "reset": ["critic", "optimizer", "iteration", "environment_step_counter"],
    }


class PolicyInitRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        checkpoint = train_cfg.get("init_policy_from")
        if checkpoint and train_cfg.get("resume"):
            raise ValueError("Choose either init_policy_from (new experiment) or resume.")
        super().__init__(env, train_cfg, log_dir, device)
        provenance = {"mode": "scratch"}
        if checkpoint:
            provenance = initialize_policy(self, Path(checkpoint))
            print(f"[INFO] Initialized actor from {provenance['checkpoint']}; fresh training state")
        if log_dir:
            path = Path(log_dir) / "params" / "initialization.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(provenance, indent=2) + "\n")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._last_iteration = self.current_learning_iteration + num_learning_iterations - 1
        self._evaluated = False
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path, infos=None):
        super().save(path, infos)
        # The final checkpoint is saved before the upstream runner closes W&B. Evaluate
        # once here so held-out metrics belong to this same run, without another SLURM job.
        if self.current_learning_iteration != getattr(self, "_last_iteration", None):
            return
        if self._evaluated:
            return
        self._evaluated = True
        self.final_evaluation(path)

    def final_evaluation(self, path) -> None:
        """Held-out test at the final checkpoint; other experiments override this."""
        from experiments.sit_free.evaluate import evaluate_policy, push_sweep

        policy = self.alg.get_policy()
        was_training = policy.training
        policy.eval()
        try:
            metrics = evaluate_policy(policy, self.device)
            sweep = push_sweep(policy, self.device)
        finally:
            policy.train(was_training)
        report = Path(path).parent / "evaluation.json"
        report.write_text(json.dumps({**metrics, "push_sweep": sweep}, indent=2) + "\n")
        it = self.current_learning_iteration
        for name, value in metrics.items():
            self.logger.writer.add_scalar(f"Evaluation/{name}", value, it)
        for push, ours, zero in zip(
            sweep["pushes_mps"],
            sweep["success_fraction"],
            sweep["zero_action_success_fraction"],
            strict=True,
        ):
            self.logger.writer.add_scalar(f"Evaluation/push_{push:g}mps/success", ours, it)
            self.logger.writer.add_scalar(f"Evaluation/push_{push:g}mps/zero_action", zero, it)
        if hasattr(self.logger.writer, "save_file"):
            self.logger.writer.save_file(str(report))
        print(f"[INFO] Held-out evaluation: {json.dumps(metrics)}")
