# Physical bike and hand-clip dimensions

Measurements are scoped by model. The values reported on 2026-11-12 apply only
to `puky_bike`, now stored at `models/puky_bike/` and selected with `--model puky_bike`
or its `--model puky_bike` alias. The earlier Sketchfab calibration reference is kept separately;
these updates do not change the green Sketchfab model, the primitive bike, or
its seated K1 pose. Use SI units in simulation configuration.

## Measurements for puky_bike only (2026-11-12)

| Quantity | Reported value | Simulation units | Status |
|---|---|---|---|
| Tyre outer diameter | 22 cm | 0.22 m | Owner's reported dimension |
| Tyre outer radius | Half the reported diameter | 0.11 m | Derived from reported diameter |
| Distance between wheel centres | 50 cm | 0.50 m | Owner's reported dimension |
| Saddle height | 42 cm | 0.42 m | Owner's reported dimension |
| Handlebar height | 53 cm | 0.53 m | Owner's reported dimension |
| Total mass | 3.5–3.9 kg | 3.5–3.9 kg | Owner's reported range; weighing method not specified |

Only `puky_bike` uses this measurement reference. Its default simulation mass is
**3.7 kg**, the range's midpoint; component mass fractions remain assumptions.
Its regenerated mesh, collision shapes, attachment sites and joint origins now
match the four dimensions above: 50 cm wheelbase, 22 cm tyre diameter, 42 cm
saddle top and 53 cm at both grip centres. Height references are ground to
saddle top and ground to grip centres. The other bikes retain their existing
dimensions and masses.

## Earlier Sketchfab calibration reference (2026-11-09)

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
| Wheelbase precision | Repeat axle-centre measurements for each bike: 50 cm for puky_bike, 56.5 cm for the Sketchfab reference |
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

The existing [primitive model](../models/primitive_bike/bike.xml) still has a 0.30 m
tyre diameter, 0.415 m saddle site, 0.61 m grip sites and 3.7 kg total mass. These
are placeholder values and do not yet match the measurements above. Its tube
inertias and mass distribution are also estimates.

The original [Sketchfab USD](../models/yellow_bike/source/SKETCHFAB_ATTRIBUTION.md) is stored at
`models/yellow_bike/source/Balance_bike.usdz`. It contains visual meshes and materials,
with no authored rigid bodies, colliders, masses or joints. Its Y-up, centimetre metadata
produces an overall model over 5 m long when composed, so verify and normalize
its scale before use.

### USDZ-derived model scaled to 56.5 cm wheelbase

The converted model is [models/yellow_bike/bike.xml](../models/yellow_bike/bike.xml).
One uniform scale makes its axle centres 0.565 m apart. A rigid rotation levels
those centres and changes Y-up to Z-up. The stem is then extended along the
steering axis to raise the grip centres to 0.45 m; the frame, saddle, wheels and
steering bearings retain their uniformly scaled geometry.
The converter defaults to these dimensions and the provisional 1.5 kg mass;
regeneration requires no dimension arguments.

| Quantity | Sketchfab calibration reference (approximate) | Current converted model |
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
[measurements.json](../models/yellow_bike/measurements.json) stores
the exact scale, source hash, derived dimensions and physical assumptions.

The four moving bodies are the frame, fork/handlebars, rear wheel and front
wheel. The visual meshes are massless and non-colliding. Compound primitives
approximate frame/fork contact, and rounded convex tyre proxies approximate
wheel contact. The model has gravity, passive wheel/steering joints and a free
root. The asset itself has no robot attachments; `k1bike.seat` adds the hand-clip
welds and seated pose in the combined scene (see "Seated K1" below). Friction, component
mass distribution, tyre behaviour and steering resistance are still provisional.

A better-looking mesh does not establish correct physical properties. Keep its
visual geometry separate from the collision, mass and inertia assumptions, and
record every calibration change against the measurements in this document.

### Real bike from the uploaded scan

The third option is [models/puky_bike/bike.xml](../models/puky_bike/bike.xml),
selected with `--model puky_bike`. Its source exports are stored under
`models/puky_bike/source/`.
Fitted tyre annuli define the axle centres and scan orientation; uniform scaling
sets the wheelbase to **0.50 m**. Each visible wheel is sized radially to **0.11 m**
maximum radius; symmetric rounded contact hulls use that radius and an inferred
**0.0496 m** width. The exposed seat post is shortened to a **0.42 m saddle top**.
The bar is levelled and the exposed stem adjusted to give **0.53 m grip-centre
height on both sides**, preserving grip spacing. Collision geometry, axle and
steering origins, attachment sites and inertias are regenerated from these
landmarks. These are model adjustments using the reported dimensions; component
shape and mechanical adjustment limits remain approximate.

Its mass is **3.7 kg**, the midpoint of the owner's
reported **3.5–3.9 kg** range. Component mass distribution and contact/steering
properties remain provisional. Spatial mesh cuts approximate the joints and can
show seams during motion. See the [rescaling process](../models/puky_bike/README.md#geometry-and-articulation)
and generated
[manifest](../models/puky_bike/measurements.json).

## Seated K1

`k1_seated_bike_scene(bike_xml=...)` (`src/k1bike/scene.py`, built by `src/k1bike/seat.py`)
seats the K1 on any bike that has a solved seated pose in `models/k1_on_bike/<bike>.json`.
The default is the real bike, `models/puky_bike`; `models/yellow_bike` (the earlier USDZ-derived
bike) has a pose too. Every bike uses the same part names (`frame`, `fork`, `saddle`, `seat`,
`left_grip`, `right_grip`), so the same code seats the K1 on each:

- **Hand clips: weld constraints.** Each clip is a MuJoCo weld between a site on
  the K1's forearm (`<side>_hand_clip`) and a site on the fork/handlebar body
  (`bike/<side>_grip_clip`). It fixes all six relative degrees of freedom, as
  specified above. Steering stays a free joint, turned by the arms through the welds.
  Forearm-to-fork contacts are excluded because the clipped forearm overlaps the
  bar's collision capsule.
- **Seat: contact only, no constraint.** The pelvis (trunk bottom cylinder) rests on
  the saddle under gravity. The robot can rise, slide or lift off, so a policy
  can push with its feet; `seat_contact()` returns the saddle's normal force for a
  "stay seated" reward or termination.
- **Collisions: no passing through.** The K1 keeps all 174 self-collision pairs and
  collides with every bike collision geom (frame tubes, saddle, fork, stem, handlebar,
  wheels). The only exclusion is each forearm against the fork, above; it
  goes once the hand model is attached. In the seated pose only the pelvis touches
  the bike. `tests/test_seat.py` checks this.
- **Assumed clip frame.** The K1 has no hands; the forearm ends 0.228 m along the
  elbow-yaw link. The clip is taken as collinear with the forearm: hand centre
  2 cm in from the tip (s = 6 cm), bar centre 4 cm further along (s = 2 cm), so the
  bar sits 2 cm beyond the forearm tip. The top/bottom offsets and orientation
  above are still unmeasured; update `seat.py` and re-solve once they are.
- **Saddle collision follows the visible saddle.** The converted bikes model the saddle
  as one flat box at the height of its highest point, but a real saddle is dished: the
  Puky's visible surface is 16 mm lower in the middle, where the rider sits. On the box,
  the K1 pressed on the saddle but visibly hovered 13-14 mm above it (4-5 mm on the yellow
  bike). Every scene now loads bikes through `load_bike()`, which ray-casts the visible
  saddle and replaces the box's collision with 12 thin convex slices that follow it
  (`src/k1bike/saddle.py`); they match the visible surface within 1 mm. The box keeps its
  mass but no longer collides (and moves to geom group 5). Collision parts under the
  saddle that would poke up through its dip are shortened: the Puky's seat-post capsule
  ended 4 mm above the dip, and the K1 sat on the post instead of the saddle; its
  collision copy now ends 5 mm below the dip. A `saddle_dip` site marks the lowest point
  of the saddle's middle, where the K1 sits. The primitive bike has no separate visible
  saddle and keeps its box.
- **Seated pose, per bike.** `scripts/solve_seated_pose.py --bike <bike>` fits the
  pelvis into the saddle's dip, feet flat 5 mm above the floor (centres at +-0.13 m, 6 cm
  in front of the dip), clip points on the bike's grip sites, horizontal forearms and a
  trunk leaning about 20 deg forward, starting from several trunk leans and keeping the
  best fit (the fit has local optima). It refuses to write a pose whose fit fails. Each
  weld's bike-side site is placed at the robot's clip site in that pose, so the welds
  start at zero error.

Holding each pose for 10 s (weight split as a share of the K1's 193 N):

| Bike | Pelvis from dip | Clip to grip | Trunk lean | Saddle | Feet | Hands | Visible gap |
|---|---|---|---|---|---|---|---|
| `puky_bike` (real bike, default) | 3 mm | < 0.01 mm | 13 deg | 86% | 10% | 4% | about 1 mm |
| `yellow_bike` | 14 mm (flat middle) | 0.03 mm | 25 deg | 67% | 23% | 9% | < 1 mm |

`tests/test_seat.py` checks on every seated bike: more than 60% on the saddle, the feet
touching, under 15% on the hands, the pelvis within 15 mm of the dip and a visible gap
under 3 mm.

The forward lean and the feet in front of the saddle matter on the real bike: its saddle
is only 7.6 cm in front of the rear axle (18 cm on `yellow_bike`). Sitting upright with the feet
behind the hips, the K1's mass drifted behind the rear wheel and pulled the bike over
backwards through the hand clips.

The weight split comes from how the pose is held (the k1_walk PD gains). With the feet
planted exactly on the floor in the target pose, the leg joints pressed them down and the
saddle took only about half the weight. With the target feet 5 mm up, the K1 settles onto the
saddle and the feet just rest on the floor, as a rider's relaxed legs do. A trained policy
sets its own split, so staying seated still has to be rewarded, not assumed.
