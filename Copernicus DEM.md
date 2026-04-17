---
aliases:
  - Copernicus
---



A ==free==, global [[Digital Elevation Model]] dataset derived from Airbus's [[WorldDEM]] dataset
- It's a processed, edited, and hydrologically conditioned version of the TanDEM-X data, downsampled from 12m to 30m (GLO-30) and 90m (GLO-90)

The editing is important: The raw interferometric DEM has artifacts over water bodies (radar phase over water is noisy), voids over steep terrain, and other issues.
- Result is ==cleaner and more usable than raw WorldDEM data==.
- Much better alternative to [[Shuttle Radar Topography Mission|SRTM]] (more recent, better accuracy, better in vegetated areas, fewer voids, better in steep terrain).

Available on AWS Open Data Registry, Copernicus Data Space Ecosystem, OpenTopography.

==For most new geospatial work, use Copernicus GLO-30==; use SRTM only if you specifically need 2000s-era terrain, or are maintaining an existing pipeline. Consider using WorldDEM if you need 12m resolution and have budget, or use [[Light Detection and Ranging|LiDAR]] where it exists and if your application justifies it.

