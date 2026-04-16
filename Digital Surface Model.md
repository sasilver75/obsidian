---
aliases:
  - DSM
---
[[Raster]] of an elevation of the top surface - includes buildings, trees, everything above bare earth.
- ==A DSM is a [[Digital Elevation Model|DEM]] of a surface== (including buildings, trees, powerlines)

In practice, raw [[Light Detection and Ranging|LiDAR]] or [[Photogrammetry]] gives you a [[Digital Surface Model|DSM]] first.
- You then classify the point cloud (separating ground points from above-ground returns) to produce a [[Digital Terrain Model|DTM]].
- The difference between DSM and DTL at any pixel is essentially the height of whatever is sitting on the ground there.