# Orange hand clips

The K1 has no hands. These add-on clips attach each forearm to a handlebar
grip. The code is `src/k1bike/hand_clips.py`: it adds the clip geometry, hand/wrist sites
and fixed hand attachments to a K1 + bike scene.

- Total length: 10 cm.
- Hand/handle centre spacing: 4 cm, at equal height.
- Hand-side nominal diameter: 4 cm; handle side: about 3 cm or less.

```python
from k1bike.hand_clips import add_orange_clips, lock_fitted_clips
```

Call `add_orange_clips(spec)` before compiling. Fit the hands to the target sites,
then call `lock_fitted_clips(spec, model, data)`.

Clip shape and hand frames are approximate. [Measurements](../../docs/bike-dimensions.md#passive-hand-clips).

**Not used by the seated scene yet.** `k1_seated_bike_scene()` still welds an assumed clip
point on each forearm to the grip ([k1_on_bike](../k1_on_bike/README.md)) and excludes
forearm-fork contacts. Switching it to these clips means re-solving the seated pose and
removing that exclusion, so the clips themselves touch the bar.
