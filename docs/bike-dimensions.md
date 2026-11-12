# Physical bike and hand-clip dimensions

Recorded on 2026-11-09 from the project owner's approximate measurements and
estimate of the physical balance bike. Use SI units in simulation configuration.
These values describe the physical bike, independently of the downloaded visual
asset or the current placeholder simulation model.

## Available measurements

| Quantity | Reported value | Simulation units | Status |
|---|---|---|---|
| Tyre outer diameter | About 25 cm | About 0.25 m | Approximate measurement |
| Tyre outer radius | Half the reported diameter | About 0.125 m | Derived from approximate diameter |
| Distance between wheel centres | About 56.5 cm | About 0.565 m | Approximate measurement; current uniform-scaling reference |
| Saddle height | About 34–35 cm | About 0.34–0.35 m | Approximate measurement |
| Handlebar height | About 45 cm | About 0.45 m | Approximate measurement |
| Total bike mass | Roughly 1–2 kg | Roughly 1–2 kg | Owner's estimate; bike has not been weighed |

The mass range is a working estimate, not a measured bound. The USDZ-derived
prototype currently uses **1.5 kg** as a provisional point within that estimate.
Its component mass fractions are also assumptions, not measurements. Reweigh
and recalibrate before treating either as the physical bike's mass properties.

For repeatable height measurements, confirm the reference points: level ground
to the saddle's top surface, and level ground to the handlebar grip centres.
The reference points used for the reported heights have not yet been confirmed.
The diameter should be the outside of the inflated tyre, not the rim diameter.

## Passive hand clips

The orange plastic clips mechanically retain the hands on the handlebars and
have no motors. The owner clarified that the handlebar grips have elliptical
cross-sections that prevent the proposed relative rotation. Model each clip as
a **fixed hand-to-handlebar attachment**: constrain relative translation and
rotation. This supersedes the earlier proposal for a passive x-axis hinge.

The following dimensions were reported on 2026-11-09. The owner confirmed that
the 2 cm and 6 cm centre positions are measured from the same end. Use a ruler
coordinate `s` along the clip length, with `s = 0` at that end. Its orientation
relative to each robot link and the handlebars still needs to be established.

| Quantity | Reported value | Simulation units | Status |
|---|---|---|---|
| Total clip length | 10 cm | 0.10 m | Reported dimension |
| Bike-side feature centre | At 2 cm | `s = 0.02 m` | Centre position and common reference end confirmed |
| Start of robot-hand section | At 4 cm | `s = 0.04 m` | Reported position |
| Robot-hand feature centre | At 6 cm | `s = 0.06 m` | Centre position and common reference end confirmed |
| Start of wrist section | At 8 cm | `s = 0.08 m` | Reported position |
| Hand-side clip diameter | 4 cm | 0.04 m | Applies to the hand side; outside versus opening diameter still to confirm |
| Handlebar-side size | About 3 cm or less | Roughly 0.03 m or less | Elliptical section; separate major/minor dimensions not measured |

Derived dimensions:

- The bike-side and robot-hand centres are separated by `6 - 2 = 4 cm` (0.04 m).
- The robot-hand section extends from 4 to 8 cm, giving a 4 cm length.
- The final wrist section extends from 8 to 10 cm, giving a 2 cm length.
- The hand-side 4 cm diameter corresponds to a 2 cm radius (0.02 m).

The two clip features are on opposite top/bottom sides. Which feature is on top,
the directions of the openings, and any transverse offsets have not yet been
specified. Do not assume that the two interfaces have the same orientation.
The owner later clarified that the mounted hand and handlebar centres are at
approximately the same height. Do not interpret their 4 cm separation as a
vertical offset; its horizontal direction and attachment frames still need
confirmation.

Use the 4 cm centre separation for the offset within the clip rather than its
full 10 cm length. This does not by itself specify an offset from the robot's
wrist-link origin: the wrist transition begins at `s = 0.08 m`, and the link's
attachment frame must be located separately.

For simulation, align the measured hand and handlebar attachment frames before
enabling a fixed-pose constraint (a weld in MuJoCo). Connect each hand to the
moving handlebar/fork body. The bike's steering joint remains free, and the bike
and robot remain free to lean and fall together. Clip strength, flexibility and
breakage are not measured; a fixed constraint is the initial idealization.

## Still to measure

| Quantity | Measurement or definition |
|---|---|
| Wheelbase precision | Repeat the approximate 56.5 cm measurement between axle centres |
| Tyre width | Outside width of the inflated tyre |
| Handlebar width | Distance between the intended left and right clip attachment points |
| Saddle position | Horizontal distance from the rear axle to the saddle reference point |
| Steering geometry | Steering-axis angle and fork offset or trail |
| Steering limits | Maximum usable left/right steering angles |
| Clip attachment frames | Establish the top/bottom mapping, transverse offsets and orientation relative to the robot's wrist and handlebars |
| Handlebar ellipse and clip clearance | Measure both ellipse axes and distinguish opening dimensions from outside plastic dimensions |
| Total mass | Weigh the bike; exclude the robot and record whether clips are included |
| Mass distribution | Component masses or centre-of-mass estimates, with the method recorded |

Friction, tyre compliance, wheel rolling resistance and steering resistance also
need physical experiments; dimensions alone do not determine those parameters.

## Current models and downloaded asset

The existing [primitive model](../assets/bike/balance_bike.xml) still has a 0.30 m
tyre diameter, 0.415 m saddle site, 0.61 m grip sites and 3.7 kg total mass. These
are placeholder values and do not yet match the measurements above. Its tube
inertias and mass distribution are also estimates.

The original [Sketchfab USD](../assets/bike/SKETCHFAB_ATTRIBUTION.md) is stored at
`assets/bike/Balance_bike.usdz`. It contains visual meshes and materials,
with no authored rigid bodies, colliders, masses or joints. Its Y-up, centimetre metadata
produces an overall model over 5 m long when composed, so verify and normalize
its scale before use.

### USDZ-derived model scaled to 56.5 cm wheelbase

The converted model is [sketchfab_mujoco/bike.xml](../assets/bike/sketchfab_mujoco/bike.xml).
One uniform scale makes its axle centres 0.565 m apart. A rigid rotation levels
those centres and changes Y-up to Z-up. The stem is then extended along the
steering axis to raise the grip centres to 0.45 m; the frame, saddle, wheels and
steering bearings retain their uniformly scaled geometry.
The converter defaults to these dimensions and the provisional 1.5 kg mass;
regeneration requires no dimension arguments.

| Quantity | Physical bike (approximate) | Current converted model |
|---|---|---|
| Wheel-centre distance | 56.5 cm | 56.50 cm |
| Outside tyre diameter | 25 cm | 25.73 cm |
| Saddle top height | 34–35 cm | 35.83 cm |
| Grip centre height | 45 cm (reference point to confirm) | 45.00 cm, after stem extension |
| Tyre width | Not measured | 4.24 cm, inferred from visual asset |
| Left/right grip centre spacing | Not measured | 26.97 cm, inferred from visual asset |

Uniform scaling alone gives a 41.01 cm grip-centre height. The owner requested
raising the handlebars: the derivative now adds **4.30 cm of stem length** along
the inclined steering axis, moving the complete bar and grips **3.99 cm upward
and 1.59 cm rearward**. Only the short stem section above the headset is
lengthened; bar width, grip shape and the steering joint remain unchanged.
This is a requested geometric adjustment, not a measured stem setting or proof
of its physical adjustment range. The saddle remains at 35.83 cm. The generated
[measurements.json](../assets/bike/sketchfab_mujoco/measurements.json) stores
the exact scale, source hash, derived dimensions and physical assumptions.

The four moving bodies are the frame, fork/handlebars, rear wheel and front
wheel. The visual meshes are massless and non-colliding. Compound primitives
approximate frame/fork contact, and rounded convex tyre proxies approximate
wheel contact. The model has gravity, passive wheel/steering joints and a free
root; the hand clips and seated robot are not attached yet. Friction, component
mass distribution, tyre behaviour and steering resistance are still provisional.

A better-looking mesh does not establish correct physical properties. Keep its
visual geometry separate from the collision, mass and inertia assumptions, and
record every calibration change against the measurements in this document.

### Real bike from the uploaded scan

The third option is [real_bike/bike.xml](../assets/bike/real_bike/bike.xml),
selected with `--model real_bike`. Its source exports are stored under
`assets/bike/real_bike/source/`.
Fitted tyre annuli define the axle centres and scan orientation; uniform scaling
sets the wheelbase to 0.565 m. A separate symmetric contact model uses 0.125 m
tyre radius and an inferred 0.056 m width.

Preserving the scanned geometry gives a **0.5044 m saddle top** and **0.5945 m
left / 0.5830 m right grip centres**. These are derived scan dimensions, not new
physical measurements, and differ substantially from the reported heights.
The scan retains its asymmetry. It has not been adjusted to the measured seat
or handlebar settings. Its 1.5 kg mass and all contact/steering properties remain
provisional. Spatial mesh cuts approximate the joints and can show seams during
motion. See the [scan notes](../assets/bike/real_bike/README.md) and generated
[manifest](../assets/bike/real_bike/measurements.json).
