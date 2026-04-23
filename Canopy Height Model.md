---
aliases:
  - CHM
  - Canopy Height Map
---
A [[Raster]] dataset representing the height of vegetation above the ground surface, essentially a map of how tall trees and other vegetation are at every point across a landscape.

It's typically derived as the difference between two other elevation products:

[[Canopy Height Model|CHM]] = [[Digital Surface Model|DSM]] - [[Digital Terrain Model|DTM]]

Where:
- DSM = Elevation of the top of everything
- DTM = Elevation of bare earth only

Data sources:
- [[Light Detection and Ranging|LiDAR]] (gold standard)
- [[Photogrammetry]] (SfM)
- [[Satellite]] stereo
- [[Global Ecosystem Dynamics Investigation]] (GEDI)
