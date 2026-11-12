# Real bike for MuJoCo

This third bike option derives from [`source/real_bike.obj`](source/real_bike.obj).
The source OBJ, material, colour/AO/normal textures and companion USDZ are stored
in [`source/`](source/). The exports were renamed and their material/texture
references updated; source geometry, UVs, texture images and USDZ are preserved.
The viewer selects this derivative with `--model real_bike`; the default remains
the green Sketchfab bike. Both the bike-only and K1/bike scenes support it.
See the [platform-specific viewer commands](../../../docs/commands.md#view-the-bike-or-the-k1-and-bike).

## Geometry and articulation

The converter fits the two tyre annuli, rotates the oblique scan into x-forward,
y-left, z-up coordinates, and uniformly scales the axle spacing to **0.50 m**.
The origin is on the floor below the rear axle. All **25,000 source triangles**
and their UV coordinates are retained. Four generated OBJ meshes split the scan
into the frame, fork/handlebars, rear wheel and front wheel. MuJoCo uses the
original colour texture; the source AO and tangent-space normal maps are unused.
Keep the source texture at its existing path: the MJCF references it relatively.

The scan is fused and has no authored mechanical parts. Spatial cuts near the
tyres and steering bearings approximate those boundaries. Small gaps, rough
edges and misassigned boundary triangles can appear during articulation; this
is not a clean mechanical reconstruction. The complete outer head tube stays
on the frame while the stem, handlebars and lower fork steer together.

The owner's measurements apply only to this model. The generated geometry and
physics now use:

| Quantity | Value |
|---|---|
| Axle-centre distance | 50 cm |
| Outer tyre diameter | 22 cm |
| Saddle top above ground | 42 cm |
| Left and right grip centres above ground | 53 cm |
| Total mass | 3.7 kg, midpoint of the reported 3.5–3.9 kg range |

The converter achieves these dimensions in four steps:

1. Uniformly scale the initial 56.5 cm-wheelbase geometry by `50 / 56.5 = 0.884956`.
   Position the wheel axles 11 cm above the floor, adjusting the vertical origin.
2. Size each wheel's visible mesh radially to 11 cm maximum radius about its axle.
   Generate matching rounded collision hulls with the same radius.
3. Lower the saddle along its seat-post axis to a 42 cm top height. Relative to
   uniform scaling, it moves 2.576 cm down and 0.826 cm forward. Blend this
   displacement through the exposed post while keeping its lower clamp fixed.
4. Level the complete handlebar by removing approximately 2° of scan roll,
   preserving grip spacing. Move the bar and grips 0.960 cm up and 0.320 cm
   rearward along the steering axis; blend the adjustment through the exposed
   stem. Both grip centres then sit at 53 cm.

The saddle and bar retain their shapes, with local deformation confined to the
exposed posts. Lower frame and headset geometry retain the uniform scale.
Regenerate the collision proxies, grip/seat sites, axle and steering origins,
and inertias from the resulting landmarks. These adjustments describe the
derivative and do not establish the real bike's mechanical adjustment limits.
See [the measurement reference](../../../docs/bike-dimensions.md).

## Provisional physics

The derivative adds a free root, passive steering and wheel hinges, gravity,
rounded tyre contact hulls, compound frame/fork colliders, and `seat`,
`left_grip`, `right_grip` sites. Visuals have zero mass and no collision in group
2; collision proxies use group 3. There is no balancing controller or robot
attachment. The K1 scene places the robot beside the bike.

The default collision tyre diameter is **0.22 m**;
its inferred width is approximately **0.0496 m**. These are idealized symmetric colliders rather
than the scan's irregular tyre surfaces. Total mass is **3.7 kg**, the midpoint
of the owner's reported **3.5–3.9 kg** range for this bike. Component mass ratios
still follow the original prototype and remain provisional. Wheel inertia assumes a
uniform torus. Steering geometry, limits, friction and damping are estimates.
Passing simulation checks does not calibrate these parameters to the real bike.

[`measurements.json`](measurements.json) records source hashes, alignment,
dimensions, face assignments and assumptions. The source was supplied by the
project owner; no external author or licence metadata was supplied with it.

## Reproduction

**Linux and macOS**, from the repository root, headless with Python 3.12:

```bash
uv run python scripts/convert_real_bike.py
```

No extra conversion package is required. Optional `--wheelbase`, `--tyre-diameter`,
`--saddle-height` and `--handlebar-height` set the four dimensions independently,
in metres. `--mass` changes the total mass in kilograms. Wheelbase sets the
baseline uniform scale; wheel sizing and post adjustments then meet the other
targets. Friction and component mass distribution remain provisional.
`--source` expects this inspected OBJ and its companion colour texture, and
rejects a different OBJ hash. Changing the source requires inspecting its
alignment and spatial cuts again.
