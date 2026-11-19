"""Export the free-bike policy in the same deployment layout as the supported sit policy.

Linux host, repository root:
    uv run --group train python -m experiments.sit_free.export --checkpoint CHECKPOINT.pt --out policy.pt --device cpu
"""

from experiments.sit_free.task import TASK_ID
from experiments.walk.export import main

if __name__ == "__main__":
    main(task_id=TASK_ID)
