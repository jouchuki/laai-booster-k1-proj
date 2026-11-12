# Orange hand clips

`orange_clips.py` adds clip geometry, hand/wrist sites and fixed hand attachments.

- Total length: 10 cm.
- Hand/handle centre spacing: 4 cm, at equal height.
- Hand-side nominal diameter: 4 cm; handle side: about 3 cm or less.

Call `add_orange_clips(spec)` before compiling. Fit the hands to the target sites,
then call `lock_fitted_clips(spec, model, data)`.

Clip shape and hand frames are approximate. [Measurements](../../docs/bike-dimensions.md#passive-hand-clips).
