"""Save joystick route and direction metrics with the final checkpoint."""

import json
from pathlib import Path

from experiments.sit_free.runner import PolicyInitRunner


class JoystickRunner(PolicyInitRunner):
    def final_evaluation(self, path) -> None:
        from experiments.joystick.evaluate import evaluate_policy

        policy = self.alg.get_policy()
        was_training = policy.training
        policy.eval()
        try:
            report = evaluate_policy(policy, self.device)
        finally:
            policy.train(was_training)
        file = Path(path).parent / "evaluation.json"
        file.write_text(json.dumps(report, indent=2) + "\n")
        writer = self.logger.writer
        for group in ("route", "forward", "reverse"):
            for name, value in report[group].items():
                if value is not None:
                    writer.add_scalar(
                        f"Evaluation/{group}/{name}", value, self.current_learning_iteration
                    )
        if hasattr(writer, "save_file"):
            writer.save_file(str(file))
        print(f"[INFO] Joystick evaluation: {json.dumps(report)}")
