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
y-left, z-up coordinates, and uniformly scales the axle spacing to **0.565 m**.
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

Uniform scaling preserves the scanned saddle and handlebar positions. It gives
a **50.44 cm saddle top** and approximately **58.30/59.45 cm grip-centre heights**,
including the scan's asymmetry. These differ from the owner's approximate
34–35 cm saddle and 45 cm handlebar measurements. No seat-post shortening or
stem adjustment is applied. See [the measurement reference](../../../docs/bike-dimensions.md)
before using this geometry for robot seating or hand attachments.

## Provisional physics

The derivative adds a free root, passive steering and wheel hinges, gravity,
rounded tyre contact hulls, compound frame/fork colliders, and `seat`,
`left_grip`, `right_grip` sites. Visuals have zero mass and no collision in group
2; collision proxies use group 3. There is no balancing controller or robot
attachment. The K1 scene places the robot beside the bike.

The default collision tyre diameter is the approximately measured **0.25 m**;
its inferred width is **0.056 m**. These are idealized symmetric colliders rather
than the scan's irregular tyre surfaces. Total mass is a provisional **1.5 kg**,
with the original prototype's component mass ratios. Wheel inertia assumes a
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

No extra conversion package is required. Optional `--wheelbase` uniformly scales
geometry and collision proxies; `--mass` changes the provisional total mass.
`--source` expects this inspected OBJ and its companion colour texture, and
rejects a different OBJ hash. Changing the source requires inspecting its
alignment and spatial cuts again.
