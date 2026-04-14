An open-standard spatial extension to the Apache [[Apache Parquet|Parquet]] file format, adding geospatial data types line points/lines/polygons.

The geometry is stored as [[Well-Known Binary|WKB]] bytes in a column, with metadata describing both the [[Coordinate Reference System|Coordinate Reference System]] and bounding box.

Advantage:
- Get Parquet's columnar analytics performance on geospatial data.
	- [[DuckDB]] with its spatial extension can query a 10GB GeoParquet file with spatial filters in seconds without loading it into memory.

Often seen for large open geospatial datasets distributed for analytics:
- Overture Maps
- Microsoft Building Footprints
- Google Open Buildings
... and any dataset where you want analytics-first access rather than map rendering.