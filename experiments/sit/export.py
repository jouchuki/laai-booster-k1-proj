"""Export a trained sit policy for booster_deploy, check it, and record a video.

Same format and checks as the walk export (experiments/walk/export.py): 690 inputs in
booster_deploy's layout, 20 actions around the seated pose, the velocity command input zero.

Linux GPU cluster node or NVIDIA host (from the repository root):
    uv run --group train python -m experiments.sit.export --checkpoint logs/laai_sit/<run>/model_1499.pt
    MUJOCO_GL=egl uv run --group train python -m experiments.sit.export --checkpoint ... --video sit.mp4

Keep policy.pt as experiments/sit/runs/runN/policy.pt.
"""

from experiments.sit.task import TASK_ID
from experiments.walk.export import main

if __name__ == "__main__":
    main(task_id=TASK_ID)
