"""Training tasks, one module per phase of the plan in the README.

Importing this package registers every task with mjlab.

    walk.py           phase 1  K1 velocity-tracking walk on flat ground   (K1Bike-Walk-Flat)
Planned:
    sit_supported.py  phase 3  sit on a held-upright bike, hands on grips, feet down
    push_supported.py phase 4  push off with alternating feet, bike held upright
    ride_free.py      phase 5  bike released; balance by steering, catch with a foot
"""

from k1bike.tasks import walk  # noqa: F401
