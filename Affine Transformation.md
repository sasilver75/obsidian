A linear mapping that ==preserves straight lines and ratios of distances, but not necessarily angles or lengths.==

It's the mathematical tool for converting between coordinate systems when the relationship is linear but not necessarily just a simple scale.

Any affine transformation in 2D is some combination of:
- Translation (shift origin)
- Scale (shrink/stretch axes)
- Rotation (around a point)
- Shear (Skewing along an axis)
- Reflection (Flipping across an access)

It CANNOT do perspective distortion, curves, or non-linear wwarping.

The 6 parameters (a, b, c, d, tx, ty) fully define the
  transform:
  - tx, ty — translation (the origin/offset)
  - a, d — scale in X and Y
  - b, c — rotation and/or shear


Why affine and not something more complex?

For small geographic areas, the Earth is locally flat enough that pixel→ground can be treated as linear. For large areas or high-accuracy work, you'd need a proper map projection (which is a non-linear operation) applied first, and then an affine transform maps from that projected coordinate space to pixels. GeoTIFFs store the affine part; the CRS tag handles the projection.