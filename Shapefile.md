A legacy format for geospatial feature data from [[ESRI]], introduced in the early 1990s.
- Prior to things like Shapefile, each specific government agency had its own file format and file type for geospatial data, having very little compatibility. Then ESRI came on to introduce the Shapefile, which was a non-topological file format for vector data that quickly gained adoption.

A "shapefile" is actually ==3-7 files== that must travel together:
- `.shp` (contains Geometry)
- `.dbf` (Feature atribute table (dBASE format))
- `.prj` (Projection/[[Coordinate Reference System|CRS]] definition)
- `shx` (Shape index, linking the geometry to hte attributes)

You'll encounter these constantly when downloading from government GIS portals!
In Python, `geopandas` can read them natively, but likely convert to GeoJSON or load into PostGIS before doing anything serious.

