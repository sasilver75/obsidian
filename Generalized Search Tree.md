---
aliases:
  - SP-GiST
  - Space-Partitioned GiST
  - GiST
---
A flexible indexing method in [[PostgreSQL|Postgres]] supporting various data types and queries, particularly useful for complex data like ==geometry== and text search.

It's not a single algorithm, it's a framework that lets extension authors plug in custom index logic. [[PostGIS]] for instance implements an [[R-Tree]] inside GiST for geometry types.

```sql
-- Standard syntax for a PostGIS geometry index
CREATE INDEX idx_sr_311_geom ON public.sr_311 USING GIST (geom);
```
- Above: "USING GIST" tells PostgreSQL to use the GiST index method.

This GiST implementation stores bounding boxes and supports operators like &&" for BBox overlap and "<->" for distance:
```sql
-- Bounding box intersection (fast — uses index)
SELECT * FROM sr_311
WHERE geom && ST_MakeEnvelope(-118.35, 34.0, -118.25, 34.1, 4326);

-- Points within 500m of a location (fast — uses index via ST_DWithin)
SELECT * FROM sr_311
WHERE ST_DWithin(geom::geography, ST_MakePoint(-118.25, 34.05)::geography, 500);

-- Which neighborhood contains this point (fast — GiST on both tables)
SELECT n.name FROM la_neighborhoods n
WHERE ST_Contains(n.geom, ST_MakePoint(-118.25, 34.05));
```


### GiST is approximate.
GiST indexes bounding boxes, not exact geometry. A query first uses the index to find candidates whose bounding boxes match, then applies the exact geometry function as a filter. 

Two steps:
1. Index Scan: "Which *bounding boxes* overlap the search area?" (using tree-like structure of Bboxes, which are all larger than the geometries they contain)
2. Recheck: "For these candidates, which ones actually satisfy the exact condition?" (exact answer, applying the expensive check to a filtered subset of geometries from step 1)


### SP-GiST, for H3 and Recursive Structures
 - Postgres also has an [[Generalized Search Tree|SP-GiST]] index, which implements space-partitioning trees like [[QuadTree]]s and [[k-d Trees]].
	 - This works better than GiST for data that naturally partitions space evenly, like [[H3]] cells, which tile space perfectly.

> Aside: In our LA Observatory project, we used [[B-Tree]] indexes on H3 columns, rather than GiST/SP-GiST, because we query H3 by exact cell ID, not by proximity.


