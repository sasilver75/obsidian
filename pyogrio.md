Python library providing fast, bulk-oriented read/write access to [[Geospatial Data Abstraction Library|GDAL]]/[[Geospatial Data Abstraction Library|OGR]] vector data sources, such as [[Shapefile]], [[GeoPackage]], [[GeoJSON]], and several others.
- These vector data sources typically have geometries and associated records with potentially many columns.

The typical use of `pyogrio` is to read or write these data sources to or from [[Geopandas]] `GeoDataFrames`
- Indeed, Geopandas is "built on" pyogrio