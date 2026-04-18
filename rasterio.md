Pronounced "Rasteereo", rhymes with Cheerio.

==The primary Python library for reading and writing [[Raster]] geospatial data:==
- Satellite imagery
- Aerial photos
- Elevation models

... and any other data stored as a grid of pixel values!

==The underlying engine it wraps is "[[Geospatial Data Abstraction Library|GDAL]]"== (Geospatial Data Abstraction Library), which has existed since the late 90s and supports ~160 raster formats.
- GDAL's Python bindings are notoriously painful to use, so Rasterio wraps GDAL with a clean, Pythonic API, similar to how [[Fiona]] wraps GDAL's [[Vector]] side.

With Rasterio, you might:
1. Open a [[GeoTIFF]] (satellite image) and read pixel values as [[NumPy]] arrays.
2. Reproject rasters from one coordinate system to another
3. Clip a raster to a polygon (e.g. "give me only the elevation data within LA County")
4. Read raster metadata: [[Coordinate Reference System|CRS]], pixel size (resolution), bounding box, band count
	1. (Band Count refers to how many separate data layers a Raster file contains, where each band stores one value per pixel! RGB image has 3 bands, RGBA 4 bands, etc. Hyperspectral satellite data might have 200+ bands, each capturing a different wavelength of light!)
5. Write processed arrays back to disk as GeoTiff