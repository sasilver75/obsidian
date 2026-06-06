A [[PostgreSQL]] extension that adds a `geometry` column type and hundreds of spatial functions -- it's the standard for serious geospatial backends.

PostGIS has ==two spatial types==:
- ==Geometry==: Assumes a fast plane. Fast, but distance calculations in ESPG:4326 are in degrees, not meters -- wrong for anything beyond a tiny area.
- ==Geography==: Treats coordinates as lat/lon on a sphere. Slower, but `ST_Distance` and `ST_DWithin` return accurate meters anywhere on earth.

Rule of thumb: Store as `geometry(Point, 4326)` for compatability/indexing performance, and then cast to `::geography` in queries that need accurate distance or area calculations.


Storing geometries:
```sql
CREATE TABLE service_requests (
  id          SERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ,
  type        TEXT,
  geom        GEOMETRY(Point, 4326)  -- Point geometry in WGS84
);
```

Spatial indices: Without these, spatial queries do a full table scan:
```sql
CREATE INDEX idx_service_requests_geom
ON service_requests USING GIST (geom);
```
Above, the [[Generalized Search Tree|GiST]] index is an [[R-Tree]] spatial index.


Some PostGIS examples:
```sql
-- Create a point from lon/lat
ST_SetSRID(ST_MakePoint(-118.2437, 34.0522), 4326)

-- Distance between two points (in meters, using geography)
ST_Distance(a.geom::geography, b.geom::geography)

-- Points within X meters of a location
SELECT * FROM service_requests
WHERE ST_DWithin(
  geom::geography,
  ST_SetSRID(ST_MakePoint(-118.25, 34.05), 4326)::geography,
  500  -- meters
);

-- Which neighborhood polygon does this point fall in? (point-in-polygon)
SELECT n.name
FROM neighborhoods n, service_requests s
WHERE ST_Within(s.geom, n.geom)
AND s.id = 12345;

-- Count service requests per neighborhood
SELECT n.name, COUNT(s.id)
FROM neighborhoods n
LEFT JOIN service_requests s ON ST_Within(s.geom, n.geom)
GROUP BY n.name;

-- Convert geometry to GeoJSON for API responses
SELECT ST_AsGeoJSON(geom) FROM service_requests LIMIT 1;

-- Buffer: all points within 1km of a school
SELECT s.*
FROM service_requests s
WHERE ST_DWithin(
  s.geom::geography,
  (SELECT geom::geography FROM schools WHERE name = 'Crenshaw HS'),
  1000
);
```




