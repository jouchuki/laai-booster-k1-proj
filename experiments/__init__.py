"""One folder per experiment: a skill from the project plan (see experiments/README.md).

Importing this package registers every experiment's mjlab training task.
"""

from experiments.sit import task as sit_task  # noqa: F401
from experiments.walk import task as walk_task  # noqa: F401
