---
aliases:
  - Georeferencing
  - Direct Georeferencing
---
==Aligns digital images== or [[Raster]] datasets (like scanned maps or aerial photos) ==to a known coordinate system==, ==allowing them to be accurately positioned== on the Earth's surface.

The process typically involves identifying a sample of several [[Ground Control Point]]s (GCPs) with known locations on the image and the ground, then using curve fitting techniques to generate a parametric formula to transform the rest of the image.

Occasionally, this process has been called ==rubbersheeting==, but that term is more commonly applied to a very similar process applied to [[Vector]] GIS data.


[[Georeference|Direct Georeferencing]] is assigning real-world coordinates to sensor data (imagery, [[Light Detection and Ranging|LiDAR]]) using only the sensor's own position and orientation measurements. No [[Ground Control Point]]s required.
- "Direct georeferencing means the sensor knows where it it was and where it was looking, no ground markers needed."