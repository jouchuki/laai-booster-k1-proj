# Balance bike models

Orange hand clips: [assets/hand_clips](../hand_clips/README.md).

The viewer now defaults to the green [USDZ-derived model](sketchfab_mujoco/README.md),
uniformly scaled to the reported **56.5 cm wheel-centre distance**. It has moving
wheels and steering with provisional physics. Use `--model primitive` to view
the original blue prototype described below.

A third option, `--model real_bike`, loads the uploaded textured blue bike scan
with moving wheels and steering. See its [conversion notes](real_bike/README.md)
for approximate mesh cuts, provisional physics and the taller scanned saddle
and handlebars. The `bike` and unattached `k1_bike` scenes accept all three
models. The `k1_seated` pose is solved for the green Sketchfab model only.

## Original primitive model

`balance_bike.xml` is a placeholder made of simple shapes. Its numbers approximate a
12-inch kids' balance bike. A policy trained on wrong bike dynamics will not transfer,
so measure the real bike and update the model before serious training.

The physical bike's approximate measurements and unweighed mass estimate are
recorded separately in [docs/bike-dimensions.md](../../docs/bike-dimensions.md).
That document is the measurement reference; the table below describes the
current placeholder model.

The original downloaded visual asset is [`Balance_bike.usdz`](Balance_bike.usdz),
with attribution in [SKETCHFAB_ATTRIBUTION.md](SKETCHFAB_ATTRIBUTION.md).
It is separate from the physical model and does not already contain wheel or
steering joints.

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
ground. That is what the early training phases assume. On the USDZ-derived bike it can: the
solved seated pose has both feet flat (see [Seated K1](../../docs/bike-dimensions.md#seated-k1)).

## Sanity check

Gravity is enabled at `(0, 0, -9.81)` m/s². The bike has a free root and no
stabilizing controller. An exactly upright, symmetric initial pose can remain
upright; the placeholder's flat-sided cylindrical tyres also provide a finite
support width. This does not mean gravity is absent.

Use controlled experiments rather than judging gravity from the upright pose:

- Drop the bike from 0.3 m above its nominal pose. Before contact, its centre of
  mass should fall approximately `0.5 * 9.81 * time²` metres.
- Release it with a 5° lean and a 0.03 m vertical offset. The unsupported bike
  should topple. The offset avoids initial tyre-edge penetration of the ground.
- Give it an initial rolling speed. The viewer initializes the wheel speeds as
  well as frame velocity, so this begins with rolling instead of stationary
  wheels sliding along the floor.

The commands are listed for [Linux and macOS](../../docs/commands.md#view-the-bike-or-the-k1-and-bike).
All three models are checked for free fall, toppling and rolling. The flat-sided
tyre caveat above describes the original primitive model; the converted model
uses rounded tyre collision proxies.
Headless tests in `tests/test_models.py` check free fall, toppling in both
directions, and initial rolling contact velocity.

An unpowered bike's ability to recover while moving depends on speed, steering
geometry, inertia, contacts and tyre behaviour. Self-righting at an arbitrary
speed is not a universal acceptance test. Passing these basic checks establishes
that the simulated mechanism responds to gravity and contact; it does not
validate the physical bike's parameters. Tyre behaviour, mass distribution and
the precision of the approximate dimensions remain calibration work.
