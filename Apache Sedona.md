---
aliases:
  - Sedona
---
See also: [[SedonaDB]]
Previously known as GeoSpark, developed by Jia Yu and Mohamed Sarwat (started [[Wherobots]])
Donated to Apache Software Foundation in 2022, strong industry adoption and development

An open-source ==cluster computing system for processing large-scale spatial data==. It ==extends [[Apache Spark]] and [[Apache Flink]]== with native spatial datatypes, [[Geospatial Index]]es, and [[Spatial SQL]], enabling distributed geospatial analysis at scales that single-machine tools like [[Geopandas]] or [[PostGIS]] can't handle.
- GeoPandas is excellent but single-threaded and in-memory, so hits the wall at tens of millions of geometries.
- PostGIS scales better but is single-node with limits on parallelism.
If you have billions of GPS points, global building footprints, or continental-scale satellite-derived polygons, you need distributed compute.

See also: [[SedonaDB]], the commercial offering from Wherobots, a managed cloud service that runs Sedona without the operational complexity of managing a Spark cluster; similar to how Databricks is to Spark. They position SedonaDB as the spatial lakehouse platform, combining the lakehouse pattern with native spatial capabilities.

# Architecture
- Sedona integrates with Spark at multiple levels:
	- Spatial RDD Layer: Original API. Extends Spark's RDD.
	- Spatial SQL Layer: Modern, recommended API. Registers spatial functions in SparkSQL so you can write standard SQL with spatial extensions (e.g. ST_Contains)
	- DataFrame API: Spatial operations as DataFrame transformatinos, compatible with PySpark.


Implements the full [[Simple Features Access|OGC Simple Features]] spatial function set, compatible with PostGIS syntax (e.g. `ST_Point(...)`, `ST_Contains(...)`)

## Spatial Joins: The key operation
- Spatial joins are the most expensive and important operation at scale:
	- "For each record in table A, find all matching records in table B based on a spatial predicate."
	- Naively, this is O(nxm), with every geometry in A tested against every geometry in B.
- Sedonda implements ==optimized spatial join algorithms:==
	1. ***Partition-based spatial merge join***: partitions both datasets by space using the same spatial partitioning scheme ([[KDB-Tree]], [[R-Tree]], or [[QuadTree]]), so only geometries in the same or adjacent partitions need to be compared. Reduces the comparison space dramatically
	2. ***Broadcast join***: for small datasets, broadcast the smaller dataset to every executor and join locally. No shuffle needed.
	3. ***Index-accelerated join***: builds a local spatial index ([[R-Tree]]) on one side of the join within each partition, then probes with the other side.
- Sedona automatically chooses the join strategy based on data size and statistics.


# When to use Sedona
- Processing billions of GPS points or geometries
- Global-scale spatial joins (e.g. point-in-polygon for entire country datasets)
- Distributed satellite imagery processing across many scenes
- Already running Spark infra and want to add spatial capabilities
- Building a geospatial data lakehouse pipeline
- Processing [[Overture Maps Foundation|Overture Maps]] or [[OpenStreetMap]] at full global scale.


![[Pasted image 20260418011250.png]]




![[Pasted image 20260413112255.png]]

![[Pasted image 20260425103211.png]]
