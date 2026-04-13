While Apache Sedona is a cluster computing system for processing large-scale spatial data, *SedonaDB* is an open-source, single-node analytical database engine with geospatial as as first-class citizen, written in Rust.

I think we can think of it like [[DuckDB]] (columnar), but with geospatial in mind rather than as an extension (DuckDB Spatial).

![[Pasted image 20260412185016.png]]

Columnar in-memory datasets
- Spatial indexing
- Spatial statistics
- CRS tracking
- Arrow format and zero serialization overhead

Spatial query optimization
- Spatial-aware heuristic based optimization
- Spatial-aware cost based optimization
- Automatic disk spilling for large-scale spatial joins

Spatial query processing
- Spatial range query, KNN query, spatial join query, KNN join query
- Map algebra, NDVI, mask, zonal statistics
