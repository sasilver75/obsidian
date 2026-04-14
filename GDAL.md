---
aliases:
  - Geospatial Data Abstraction Library
---
[[GDAL]], [[GEOS]], and [[PROJ]]  are three C libraries that underpin almost all geospatial software. You rarely call them directly; they're the engine under tools like [[PostGIS]], [[Shapely]], [[QGIS]], and more.

GDAL is two things that got merged:
- ==GDAL proper==: Raster formats ([[GeoTIFF]], [[Cloud-Optimized GeoTIFF|COG]], [[HDF5]], [[NetCDF]], [[Zarr]], etc.)
- ==OGR==: Vector formats ([[Shapefile]], [[GeoJSON]], [[GeoPackage]], [[PostGIS]], etc.)

It provides a unified API so that you can write one piece of code that works across hundreds of formats.

GDAL is the ==universal translator of geospatial data!==
- Every time you convert a [[Shapefile]] to a [[GeoPackage]]
- ... or load a [[GeoTIFF]]
- ...or read from [[PostGIS]] into Python...
- ... GDAL is almost certainly involved!


`ogr2ogr` is a command line tool that converts vector formats, and it's basically GDAL.
