---
aliases:
  - WKB
---
Along with [[Well-Known Text|WKT]]s, WKBs are a serialization format for geometry objects, this time representing a shape as a binary blob.

Unlike plain-text WKTs, WKBs are of course ==not human readable==! It's a binary encoding of the same information, so it's ==more compact and faster to parse==. 
- [[PostGIS]] stores geometries internally in a variant called [[EWKB]], or "Extended WKB", which also encodes the [[EPSG Code]]/[[SRID]] so the [[Coordinate System|CRS]] travels with the geometry.

When you run the SQL
```sql
SELECT geom FROM my_table
```
in `psql`, you'll see a long hex string like `01010000020E61000000...`, which is the EWKB rendered in [[Hexadecimal]]. 
- When your Python code reads it via `geoalchemy2` or `psycopg2`, the driver deserializes that back into a `Shapely` geometry object automatically.


In practice: ==WKT is for humans and debugging, while WKB is what the database actually uses under the hood.== You rarely need to think about either directly: PostGIS and geopandas handle the serialization transparently.

