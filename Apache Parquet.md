---
aliases:
  - Parquet
---
Apache's columnar binary format for large tabular datasets.
- It's what data warehouses like [[BigQuery]], [[Snowflake]], and [[DuckDB]] use internally.
- Its columnar/column-oriented storage allows you to ==quickly compute statistics of one column across 10 million rows==, and also has:
	- ==built-in compression== (through [[Snappy]] or [[Zstandard]]),
	- schema metadata (types, nullable, etc.)
	- [[Predicate Pushdown]]: Query engines can skip row groups that don't match a filter without decompressing them.


See also: [[GeoParquet]], an open-standard spatial extension to the Apache Parquet file format, adding geospatial data types line points/lines/polygons.