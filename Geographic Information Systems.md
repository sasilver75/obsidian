---
aliases:
  - GIS
---
## Common Spatial Operations (with their [[PostGIS]])

#### Point-in-Polygon: "What neighborhood does this crime incident fall in?"
- Typically uses ST_Contains(polygon, point) or ST_Within(point, polygon)
```sql
SELECT n.name
FROM la_neighborhoods n
WHERE ST_Contains(n.geom, ST_SetSRID(ST_MakePoint(-118.25, 34.05), 4326));
```

#### Spatial Join: "Join every crime record to its neighborhood"
- Same as point-in-polygon, but in bulk.
```sql
SELECT c.*, n.name AS neighborhood
FROM crimes c
JOIN la_neighborhoods n ON ST_Within(c.geom, n.geom);
```
Note: Needs a [[GiST]] index on both geometry columns to be fast!

#### Buffer: "Find all 311 requests within 500m of a school."
```sql
SELECT s.*
FROM service_requests s
WHERE ST_DWithin(s.geom::geography, school.geom::geography, 500);
```

#### Intersection: "Clip a dataset to a specific area, like SilverLake"
```sql
SELECT * FROM service_requests
WHERE ST_Intersects(geom, (SELECT geom FROM neighborhoods WHERE name = 'Silver Lake'));
```

#### Aggregation by Area: Count, sum or average some attribute grouped by some spatial unit.
```sql
SELECT
  n.name,
  COUNT(*) AS incident_count,
  COUNT(*) / ST_Area(n.geom::geography) * 1e6 AS incidents_per_sqkm
FROM la_neighborhoods n
LEFT JOIN crimes c ON ST_Within(c.geom, n.geom)
WHERE c.date > NOW() - INTERVAL '1 year'
GROUP BY n.name, n.geom;
```



# Common Python Libraries for Geospatial work
- [[Geopandas]]: ==Pandas + geometry column==. Reads [[Shapefile]], [[GeoJSON]], [[GeoPackage]], and provides spatial joins, buffering, and aggregation in Python.
- [[Shapely]]: ==Geometry primitives== in Python: Create, manipulate, and test geometric objects. *Used under the hood* by `geopandas`.
- [[PyProj]]: ==[[Coordinate Reference System|Coordinate Reference System]] transformations==, wraps the PROJ library.
- `h3`: Python bindings for Uber's ==[[H3]] library==. Lat/Lon -> H3 index, neighbor lookups, [[K-Ring]]s.
- [[Fiona]]: ==Low-level geospatial file I/O== (reads/writes Shapefile, GeoJSON, etc.). *Used under the hood* by geopandas.
- [[rasterio]]: ==Read/write raster datasets== ([[GeoTIFF]], [[Cloud-Optimized GeoTIFF|COG]]). Access individual pixels, reproject rasters, compute band math.
- [[rio-cogeo]]: Create and validate [[Cloud-Optimized GeoTIFF]]s (==COGs==).
- [[Pyogrio]]: Fast I/O for vector formats, a ==faster drop-in backend for [[Geopandas]].==
- [[SQLAlchemy]] + [[GeoAlchemy2]]: SQLAlchemy ORM with PostGIS geometry type support.
- [[folium]]: A quick way to make Leaflet.js maps from Python; good for notebooks and exploration, but not for production.
- [[kepler.gl]]: Uber's heavy-duty geospatial visualization, has a Python/Jupiter interface; good for EDA.

# Common Pitfalls in GIS
- ==Swapped Coordinates==
	- Many datasets accidentally swap lat and lon, or have different standards. Often causes for points appearing in the ocean. Always sanity check, because `ST_IsValid()` won't catch this.
- ==Wrong [[EPSG Code|EPSG]] assumptions==
	- If a dataset provides coordinates in State Plane feet (EPSG:2229), but you treat them as if they were [[WGS84]] coordinates, your points will end up in the wrong hemisphere! 
- ==Antimeridian==
	- Polygons that cross the 180 degree meridian need special handling in [[GeoJSON]]
- ==Precision==
	- Storing 8+ decimals of lat/lon is more precision than GPS provides (8 decimals is about 1mm of accuracy). 6 decimal places (~11cm) is typically plenty. Excess precision wastes space and can mislead you about true sensor precision.
- ==NULL geometry==
	- Some records have null or zero coordinates (often (0,0), in the Gulf of Guinea); filter those out at ingest using a `WHERE latitude IS NOT NULL and latitude != 0`.




# Modern GIS Stack (mid-2025)
Video: [Desktop GIS is Dying. Here's what Replaced It](https://youtu.be/T2_RtffBiUM)
- Arrival of [[Cloud-Native Geospatial]] formats, which can sit in [[Blob Storage|Object Storage]] and be efficiently served.
![[Pasted image 20260417165951.png]]
Above (I think some of these categories are "sloppy"):
- Transform
	- [[Geospatial Data Abstraction Library|GDAL]]
	- seer.ai
	- [[dbt]]
	- [[Airbyte]]
	- BigGeo
	- [[A5]]
	- [[H3]]
- Analytics
	- [[CARTO]]
	- Fused
	- Foursquare/FSQ
	- ?
	- [[kepler.gl]]
	- Preset
	- Superset
- GIS
	- [[QGIS]]
	- [[Felt]]
	- Atlas
	- NextGIS
- AI
	- Klarety
	- Monarcha
	- Countour
	- Bunting Labs
	- aino
	- Mundi
	- AskEarth
	- GeoRetina
- Python
	- [[Geopandas]]
	- [[rasterio]]
	- TorchGeo
	- [[leafmap]]
	- GeoAI (?)
	- [[Lonboard]]
- Apps
	- [[Mapbox]]
	- [[deck.gl]]
	- [[MapLibre]]
	- GeoBase
	- Veda
- Orchestration
	- [[Apache Airflow|Airflow]]
	- [[Astronomer]]
	- [[Dagster]]
	- Kestra
	- [[Prefect]]
- Processing
	- [[Wherobots]]
	- Coiled
	- [[Databricks]]
	- [[Apache Sedona]]
	- [[Dask]]
	- [[Apache Spark]]
- OLTP
	- [[PostGIS]]
	- Crunchy Data
- OLAP
	- [[Snowflake]]
	- MotherDuck
	- [[DuckDB]]
	- [[Google BigQuery]]
	- [[Amazon Redshift]]
	- [[Trino]]
- Catalogs
	- [[Apache Iceberg]]
	- [[Delta Lake]]
	- [[Earthmover]]
	- [[DuckLake]]
- Cloud Storage
	- [[Amazon S3|S3]]
	- [[Cloudflare R2|R2]]
	- [[Google Cloud Storage|GCS]]
	- [[Azure Blob Storage]]
	- Wasabi
	- Obstore
- (Cloud-Native Geospatial) Formats
	- [[GeoParquet]]
	- [[FlatGeobuf]]
	- [[Cloud-Optimized GeoTIFF]]  (COG)
	- [[SpatioTemporal Asset Catalog]] (STAC)
	- [[Cloud-Optimized Point Cloud]] (COPC)
	- [[Zarr]]

