A legacy format for geospatial feature data from [[ESRI]].

A "shapefile" is actually ==3-7 files== that must travel together:
- `.shp` (contains Geometry)
- `.dbf` (Attribute table (dBASE format))
- `.prj` (Projection/[[Coordinate System|CRS]] definition)
- `shx` (Shape index)

You'll encounter these constantly when downloading from government GIS portals!
In Python, `geopandas` can read them natively, but likely convert to GeoJSON or load into PostGIS before doing anything serious.

