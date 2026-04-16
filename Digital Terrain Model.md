---
aliases:
  - DTM
---
[[Raster]] of bare-earth elevation with buildings/vegetation removed.
- ==A DTM is a [[Digital Elevation Model|DEM]] of the bare earth== (buildings/vegetation are filtered out)


In practice, raw [[Light Detection and Ranging|LiDAR]] or [[Photogrammetry]] gives you a [[Digital Surface Model|DSM]] first.
- You then classify the point cloud (separating ground points from above-ground returns) to produce a [[Digital Terrain Model|DTM]].
- The difference between DSM and DTL at any pixel is essentially the height of whatever is sitting on the ground there.