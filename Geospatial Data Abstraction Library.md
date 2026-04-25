---
aliases:
  - GDAL
  - OGR
---
[[Geospatial Data Abstraction Library|GDAL]], [[GEOS]], and [[PROJ]]  are three C libraries  that underpin almost all geospatial software. You rarely call them directly; they're the engine under tools like [[PostGIS]], [[Shapely]], [[QGIS]], and more.

GDAL is foundational for reading and writing both raster and vector geospatial data formats. It's the "plumbing" that much of the geospatial software world is built on.

GDAL is two things that got merged:
- ==GDAL proper==: *Raster* formats ([[GeoTIFF]], [[Cloud-Optimized GeoTIFF|COG]], [[HDF5]], [[NetCDF]], [[Zarr]], etc.)
- ==OGR==: *Vector* formats ([[Shapefile]], [[GeoJSON]], [[GeoPackage]], [[PostGIS]], etc
	- Used for converting between different vector data formats, as well as reprojecting between [[Coordinate Reference System|CRS]]s.

It provides a unified API so that you can write one piece of code that works across hundreds of formats.

GDAL is the ==universal translator of geospatial data!==
- Every time you convert a [[Shapefile]] to a [[GeoPackage]]
- ... or load a [[GeoTIFF]]
- ...or read from [[PostGIS]] into Python...
- ... GDAL is almost certainly involved!

Almost every geospatial tool uses GDAL:
- [[rasterio]] is a Python wrapper around GDAL raster
- [[Fiona]] and [[Pyogrio]] are wrappers around OGR (Vector)
- [[QGIS]], [[PostGIS]], MapServer, all use GDAL under the hood.

# Command line tools:
- `gdalinfo`: Inspect a raster file, spits out a lot of information about the file.
- `gdal_translate`: Convert between raster formats
- `gdalwarp`: Reproject/warp rasters
- `ogr2ogr` : Convert between vector formats
- `ogrinfo`: Inspect a vector file

