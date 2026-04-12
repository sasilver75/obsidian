
Raster Data, in contrast to [[Vector]] data (everything in geospatial is either vector or raster), represents continuous fields as grids of cells (pixels), where each cell holds a numeric value.
- Examples
	- Satellite imagery (each pixel = color values for patch of ground)
	- Land surface temperature (each pixel = degrees Celsius for that location)
	- Air quality index (interpolated between sensor stations across a grid)
	- Elevation (each pixel = meters above sea level)

Raster data is what you get from NASA/USGS satellite products, elevation datasets, and sensor interpolations.

The key difference between Raster and Vector data:
- Vector is precise at boundaries (a neighborhood polygon has a definitive edge), while raster is continuous (temperature blends smoothly between pixels).