
A general dataframe toolkit for geospatial data in [[Pandas]]/JupyterNotebook ecosystem.
Extends pandas to support spatial data through use of a Geometry column.

A `GeoDataFrame` is just a pandas `DataFrame` with a special geometry column containing [[Shapely]] geometry objects (points, lines, polygons).

```python
import geopandas as gpd

  gdf = gpd.read_file("countries.geojson")
  gdf.head()
  # country   population   geometry
  # USA       331M         POLYGON((-125 24, -66 24, ...))
  # Canada    38M          POLYGON((-140 60, ...))

  # What it adds over pandas
  gdf.crs                        # the CRS
  gdf.to_crs("EPSG:4326")        # reproject
  gdf.area                       # area of each geometry
  gdf.buffer(1000)               # 1000-unit buffer around each geometry
  gdf.centroid                   # centroid of each geometry

  # Spatial join — attach attributes based on which polygon each point falls in
  gpd.sjoin(points_gdf, polygons_gdf, predicate="within")

  # Set operations
  gpd.overlay(gdf_a, gdf_b, how="intersection")

  gdf.plot()                     # quick map visualization
```
Built on:
- [[Pandas]] (for tabular operations)
- [[Shapely]] (for in-memory geometry operations)
- [[Fiona]] and [[Pyogrio]] (for file I/O )
	- (Two separate file I/O backends that Geopandas can use; Fiona is the original and pyogrio is the newer, faster backend. Both wrap [[Geospatial Data Abstraction Library|GDAL]]/[[Geospatial Data Abstraction Library|OGR]], but pyogrio has vectorized I/O, avoiding the Python object-per-feature overhead that makes Fiona slow on large files.)
	- As of GeoPandas 10, pyogrio is the default; Fiona is being phased out.
- [[PyProj]] for [[Coordinate Reference System|CRS]] reprojection
- For very large dataset, [[Dask]]-Geopandas


A Typical workflow:
- Load vector data into [[Geopandas]]
- Load raster data into [[rioxarray]]
- Use `rio.clip(gdf.geometry)` to mask the raster to your vector boundaries.



![[Pasted image 20260425014934.png]]




