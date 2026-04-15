
Raster Data, in contrast to [[Vector]] data (everything in geospatial is either vector or raster), represents ==continuous fields as grids of cells (pixels), where each cell holds a numeric value==.
- Examples
	- Satellite imagery (each pixel = color values for patch of ground)
	- Land surface temperature (each pixel = degrees Celsius for that location)
	- Air quality index (interpolated between sensor stations across a grid)
	- Elevation (each pixel = meters above sea level)

Raster data is what you get from NASA/USGS satellite products, elevation datasets, and sensor interpolations.

The key difference between Raster and Vector data:
- Vector is precise at boundaries (a neighborhood polygon has a definitive edge), while raster is continuous (temperature blends smoothly between pixels).

In Raster data, ==Band Count== refers to how many separate data layers a Raster file contains, where each band stores one value per pixel! RGB image has 3 bands, RGBA 4 bands, etc. Hyperspectral satellite data might have 200+ bands, each capturing a different wavelength of light!


## Raster Formats
- [[GeoTIFF]]
	- The ==standard raster format==. 
	- A regular [[Tagged Image File Format|TIFF]] image file with embedded geospatial metadata:
		- What [[Coordinate Reference System|Coordinate Reference System]] the pixels are in
		- What the real-world coordinates of the image corners are.
	- A satellite image of LA is a GeoTIFF. An elevation model of the Santa Monica Mountains is a GeoTIFF. A land-use classification grid is a GeoTIFF.
	- Each "==band==" is a grid of values a standard RGB satellite image has 3 bands, an elevation model has 1 band, and multispectral satellite images may have many dozens or hundreds.
	- ==Problem==:
		- ==A classic GeoTIFF stores pixels in row order.==
		- To display a web map ==tile== at a specific bounding box and zoom level (XYZ tile), you'd need to decompress the entire image, then extract the relevant pixels. This is too slow for on-demand tile serving.
- [[Cloud-Optimized GeoTIFF]]
	- A GeoTIFF that's been ==reorganized internally for efficient HTTP range requests!==
	- Key differences:
		- ==Overview pyramids== inside the file
			- Pre-downsampled versions of the image at multiple zoom levels (like a tile pyramid), all stored inside the same file. 
		- ==Tiled internally==
			- Pixels are stored in square tiles (typically 256x256 or 512x512), instead of row by row.
		- ==Index at the start==
			- A small header listing where every tile and overview level lives in the file.
	- With this structure, a tile server like [[TiTiler]] can:
		- Fetch the header to find where zoom-level-8 tiles are
		- Issue a range request for exactly the bytes of the specific 256x256 file you need
		- Return it without touching the rest of your file
	- A 5GB satellite image COG stored on S3 can serve web map tiles without ever loading the full image into memory!
	- Satellite imagery products (e.g. [[Sentinel|Sentinel-2]], [[Landsat]]), elevation models, processed raster outputs from climate/environmental datasets will use this. Also any raster that you'd serve as a tile layer.
- [[Zarr]]
	- A format for chunked, compressed N-dimensional arrays. Think "Numpy arrays stored in a way that works well in the cloud."
	- Unlike COG (which is for 2D grids), ==Zarr handles arbitrary dimensions==: (space x time x variable), which makes it the ==dominant format for climate science and earth observation time series!==
	- ==Not a single file!== It's a directory (or object storage prefix 😜) of `.zarr` files organized as a key-value store. Each chunk of the array is a separate file.
	- You'll see it for [[NOAA]]/[[National Aeronautics and Space Administration|NASA]] climate datasets, ERA5 reanalysis, weather model outputs, any "datacube" product.