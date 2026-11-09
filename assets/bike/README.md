# Balance bike model

`balance_bike.xml` is a placeholder made of simple shapes. Its numbers approximate a
12-inch kids' balance bike. A policy trained on wrong bike dynamics will not transfer,
so measure the real bike and update the model before serious training.

## Measurements to take

| Quantity | Placeholder | Measured | Where in the XML |
|---|---|---|---|
| Wheel radius | 0.15 m | | `class="wheel"` geom size |
| Tire width | 0.04 m | | `class="wheel"` geom size (half-width 0.02) |
| Wheelbase (axle to axle) | 0.627 m | | `fork` pos + `front_wheel` pos |
| Head angle (from horizontal) | 70° | | `steer` joint axis |
| Fork offset / trail | 0 / ~5.5 cm | | `fork_tube`, `front_wheel` pos |
| Steering range | ±45° | | `steer` range |
| Seat height (ground to saddle top) | 0.415 m | | `saddle`, `seat` site |
| Handlebar height / width | ~0.62 m / 0.36 m | | `stem`, `handlebar` |
| Total mass | 3.7 kg | | geom `mass` values |
| Wheel mass | 0.6 kg each | | wheel geom `mass` |

The K1's leg is roughly 0.45 m long, so check that it can sit with both feet flat on the
ground. That is what the early training phases assume.

## Sanity check

A real bike pushed forward at a few m/s without a rider steers itself upright for a
while. Check that the model does the same before trusting it (see `scripts/view_model.py`).
