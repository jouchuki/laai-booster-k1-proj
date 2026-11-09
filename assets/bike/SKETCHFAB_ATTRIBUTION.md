# Sketchfab balance-bike source asset

- File: `Balance_bike.usdz` (original download, unchanged).
- Title: [Balance bike](https://sketchfab.com/3d-models/balance-bike-347ad1a8f70b414a95988cef24f5c8d3).
- Creator: [maulanarizal752](https://sketchfab.com/maulanarizal752).
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Source file moved from the project owner's Downloads folder on 2026-11-09.
- SHA-256: `093e92bfed8b3b044b388b08cf42c226960d785a664b8291b3a60219f076096e`.

The source contains visual meshes and materials. It has no authored physics
bodies, collisions, mass properties or joints. Keep the source file intact when
creating a simulator-ready derivative, and document any scaling, separation of
parts, or other modifications alongside that derivative.

The composed USD is Y-up with `metersPerUnit = 0.01`. Taking that metadata
literally gives a model over 5 m long, so normalize its scale against the
[physical bike dimensions](../../docs/bike-dimensions.md) before use.
