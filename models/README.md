# Models: everything the simulator loads

| Folder | What | Loaded by |
|---|---|---|
| [k1/](k1/README.md) | The Booster K1 robot (Booster's `K1_22dof.xml`, pinned, not copied here) | `k1bike.paths.k1_xml()`, `k1bike.robots.k1.k1_spec()` |
| [bike/](bike/README.md) | **The balance bike we use**: USDZ-derived, measured 56.5 cm wheelbase, provisional physics | `k1bike.paths.BIKE_XML` |
| [bike_scan/](bike_scan/README.md) | Calibrated real-bike scan: 50 cm wheelbase, 22 cm tyres, 42 cm saddle top, 53 cm grip centres, 3.7 kg | `k1bike.paths.SCAN_BIKE_XML` |
| [bike_primitive/](bike_primitive/README.md) | The first placeholder bike made of simple shapes, kept for comparison | `k1bike.paths.PRIMITIVE_BIKE_XML` |
| [hand_clips/](hand_clips/README.md) | The orange add-on hand clips that attach the forearms to the grips; not used by the seated scene yet | `k1bike.hand_clips.add_orange_clips()` |
| [k1_on_bike/](k1_on_bike/README.md) | The K1 seated on `bike/`: solved pose; scene built in code | `k1bike.scene.k1_seated_bike_scene()` |

Combined scenes (robot + bike) are built in code, not stored as XML, so they always use the
current robot and bike: `src/k1bike/scene.py` (`bike_scene`, `k1_bike_scene`,
`k1_seated_bike_scene`) and `src/k1bike/seat.py` (hand welds, seat contact).

View any of them with [the viewer commands](../docs/commands.md#view-the-bike-or-the-k1-and-bike):
`scripts/view_model.py bike | k1_bike | k1_seated`, with `--model bike_scan` or `--model bike_primitive`
for the other bikes.
The earlier `--model real_bike` option remains an alias for `bike_scan`;
`--model sketchfab` and `--model primitive` also retain their previous meanings.

## Rules

- **Physical collisions everywhere.** The K1 keeps its self-collision and collides with every
  bike collision geom (only each welded forearm against the fork is excluded). Visual meshes
  never collide; each bike's collision proxies are what the robot touches. See
  [decisions](../docs/decisions.md).
- **Measurements first.** Physical bike measurements live in
  [docs/bike-dimensions.md](../docs/bike-dimensions.md). Change a model only against them,
  and re-solve the seated pose (`scripts/solve_seated_pose.py`) after changing `bike/`.
- **Generated files are regenerated, not hand-edited**: `bike/` comes from
  `scripts/convert_bike_usdz.py`, `bike_scan/` from `scripts/convert_bike_scan.py`, `k1_on_bike/seated_pose.json` from
  `scripts/solve_seated_pose.py`.

## Sanity checks for a bike model

Gravity is enabled at `(0, 0, -9.81)` m/s². A bike has a free root and no stabilizing
controller; an exactly upright, symmetric pose can stay upright, which does not mean gravity
is absent. Check with controlled experiments instead:

- Drop the bike from 0.3 m above its nominal pose. Before contact, its centre of mass should
  fall about `0.5 * 9.81 * time²` metres.
- Release it with a 5° lean and a 0.03 m vertical offset. The unsupported bike should topple.
  The offset avoids initial tyre-edge penetration of the ground.
- Give it an initial rolling speed. The viewer sets the wheel speeds as well as the frame
  velocity, so it starts rolling instead of sliding on stationary wheels.

`tests/test_models.py` runs these checks headless on all three bikes. Passing them shows that the
mechanism responds to gravity and contact; it does not validate the real bike's parameters.
Whether an unpowered bike recovers while moving depends on speed, steering geometry, inertia,
contacts and tyres, so self-righting is not an acceptance test. Tyre behaviour, mass
distribution and the approximate dimensions remain calibration work.
