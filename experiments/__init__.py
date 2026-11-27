"""One folder per experiment: a skill from the project plan (see experiments/README.md).

Importing this package registers every experiment's mjlab training task. Without the train
dependencies (mjlab, torch), as in CI's dev-only environment, it registers nothing, so modules
that don't need them, such as experiments/push/sweep.py, still import.
"""

from importlib.util import find_spec

if find_spec("mjlab") and find_spec("torch"):
    from experiments.push import task as push_task  # noqa: F401
    from experiments.sit import task as sit_task  # noqa: F401
    from experiments.sit_free import task as sit_free_task  # noqa: F401
    from experiments.walk import task as walk_task  # noqa: F401
