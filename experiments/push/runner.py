"""The free-bike runner (actor import, fresh critic) with the push tasks' final evaluation."""

import json
from pathlib import Path

from experiments.sit_free.runner import PolicyInitRunner


class PushRunner(PolicyInitRunner):
    def final_evaluation(self, path) -> None:
        from experiments.push.evaluate import evaluate_policy

        policy = self.alg.get_policy()
        was_training = policy.training
        policy.eval()
        try:
            report = evaluate_policy(policy, self.device)
        finally:
            policy.train(was_training)
        file = Path(path).parent / "evaluation.json"
        file.write_text(json.dumps(report, indent=2) + "\n")
        it = self.current_learning_iteration
        for who in ("policy", "zero_action"):
            for row in report[who]:
                for name, value in row.items():
                    if name != "command_mps":
                        tag = f"Evaluation/{who}/{row['command_mps']:g}mps/{name}"
                        self.logger.writer.add_scalar(tag, value, it)
        if hasattr(self.logger.writer, "save_file"):
            self.logger.writer.save_file(str(file))
        print(f"[INFO] Held-out evaluation: {json.dumps(report['policy'])}")
