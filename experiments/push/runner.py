"""The free-bike runner (actor import, fresh critic) with the push tasks' final evaluation."""

import json
from pathlib import Path

from experiments.sit_free.runner import PolicyInitRunner


def row_tag(row: dict) -> str:
    """'0.2mps' for a straight row, '0.2mps_+0.1radps' for a turning one."""
    turn = row.get("command_yaw_rate_rad_s", 0.0)
    return f"{row['command_mps']:g}mps" + (f"_{turn:+g}radps" if turn else "")


class PushRunner(PolicyInitRunner):
    turns = False  # the straight tasks: speeds only

    def final_evaluation(self, path) -> None:
        from experiments.push.evaluate import EVAL_TURNS, evaluate_policy

        policy = self.alg.get_policy()
        was_training = policy.training
        policy.eval()
        try:
            report = evaluate_policy(policy, self.device, turns=EVAL_TURNS if self.turns else ())
        finally:
            policy.train(was_training)
        file = Path(path).parent / "evaluation.json"
        file.write_text(json.dumps(report, indent=2) + "\n")
        it = self.current_learning_iteration
        for who in ("policy", "zero_action"):
            for row in report[who]:
                for name, value in row.items():
                    if name not in ("command_mps", "command_yaw_rate_rad_s") and value is not None:
                        tag = f"Evaluation/{who}/{row_tag(row)}/{name}"
                        self.logger.writer.add_scalar(tag, value, it)
        if hasattr(self.logger.writer, "save_file"):
            self.logger.writer.save_file(str(file))
        print(f"[INFO] Held-out evaluation: {json.dumps(report['policy'])}")


class PushSteerRunner(PushRunner):
    turns = True  # the Steer task: speeds and turns
