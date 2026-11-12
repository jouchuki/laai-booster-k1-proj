# Primitive balance bike (old placeholder)

`bike.xml` is the first bike model, made of simple shapes (blue in the viewer). Its numbers
approximate a 12-inch kids' balance bike. It was replaced by the USDZ-derived
[yellow_bike](../yellow_bike/README.md) and then by the real bike, [puky_bike](../puky_bike/README.md), which
every scene uses by default. It is kept only for comparison:

```bash
uv run python scripts/view_model.py bike --model primitive_bike      # Linux host
uv run mjpython scripts/view_model.py bike --model primitive_bike    # macOS host
```

It has no solved seated pose, so `k1_seated` refuses it (solve one with
`scripts/solve_seated_pose.py --bike primitive_bike` if needed). Its
flat-sided cylindrical tyres give it a finite support width, so an exactly upright pose can
stay upright; see the [sanity checks](../README.md#sanity-checks-for-a-bike-model).

## Placeholder values

The physical bike's measurements are in [docs/bike-dimensions.md](../../docs/bike-dimensions.md).

| Quantity | Placeholder | Where in the XML |
|---|---|---|
| Wheel radius | 0.15 m | `class="wheel"` geom size |
| Tire width | 0.04 m | `class="wheel"` geom size (half-width 0.02) |
| Wheelbase (axle to axle) | 0.627 m | `fork` pos + `front_wheel` pos |
| Head angle (from horizontal) | 70° | `steer` joint axis |
| Fork offset / trail | 0 / ~5.5 cm | `fork_tube`, `front_wheel` pos |
| Steering range | ±45° | `steer` range |
| Seat height (ground to saddle top) | 0.415 m | `saddle`, `seat` site |
| Handlebar height / width | ~0.62 m / 0.36 m | `stem`, `handlebar` |
| Total mass | 3.7 kg | geom `mass` values |
| Wheel mass | 0.6 kg each | wheel geom `mass` |
