"""Export a push policy in the same deployment layout as the walk and sit policies.

Works for checkpoints of both push tasks: they have the same inputs, outputs and scene.

Linux host, repository root:
    uv run --group train python -m experiments.push.export --checkpoint CHECKPOINT.pt --out policy.pt --device cpu
"""

from experiments.push.task import SIMPLE_TASK_ID
from experiments.walk.export import main

if __name__ == "__main__":
    main(task_id=SIMPLE_TASK_ID)
