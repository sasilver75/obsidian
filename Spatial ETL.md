The geospatial version of a standard data pipeline (Extract, Transform, Load), but with the added complexity of geometries, projections, spatial operations, and often very large file sizes.


# Extract: Getting the Data
Spatial data comes from many places and in many formats:
- Satellite imagery from [[SpatioTemporal Asset Catalog]]s (AWS, [[Microsoft Planetary Computer|Planetary Computer]], [[Google Earth Engine|Earth Engine]])
- Vector data from APIs ([[Overture Maps Foundation|Overture Maps]], [[OpenStreetMap]], government open data portals)
- Database exports ([[PostGIS]] dumps, [[GeoPackage]]s)
- Streaming sources ([[Automatic Identification System|AIS]] ship tracking, [[Global Positioning System|GPS]] telemetry)
- FTP servers (still surprisingly common in government GIS)

The extraction layer has to handle format diversity ([[Shapefile]], [[GeoJSON]], [[GeoPackage]], [[FlatGeobuf]], [[Cloud-Optimized GeoTIFF|COG]], etc.), which are where tools like [[GDAL]]/OGR earn their keep (they're almost always the underlying reader, regardless of what higher-level tool you're using).


# Transform: The Interesting  Part
Common transform steps include:
- ==Projection Handling==: Almost always necessary; Source data arrives in inconsistent [[Coordinate Reference System]]s; You need to reproject everything to a common CRS before any spatial operation, and you should choose that CRS carefully!
	- [[Mercator|Web Mercator]] ([[EPSG Code|EPSG]]:3857) for display
	- A local [[Universal Transverse Mercator|UTM]] zone for accurate distance/area calculations
	- [[WGS84]] for storage
- ==Geometry fixing==: Real-world data is dirty. Self-intersecting polygons, unclosed rings, duplicate vertices, slivers, wrong [[Winding Order]]. 
	- ST_MakeValid in PostGIS and buffer(0) tricks in Shapely are common fixes.
	- Skipping this step causes silent failures downstream.
- ==Spatial joins==: Enriching features with attributes from another dataset. 
	- "Which census tract does each point fall in?"
	- "Which watershed does each river segment belong to?"
	- This can be expensive at scale, and will ned careful indexing.
- ==Clipping to areas of interest==: Global datasets can be clipped to the area of interest. 
	- Sounds simple, but can introduce edge effects at boundaries if you're not careful.
- ==Aggregation==: Dissolving features, zonal statistics, rolling up point data to grid cells.
	- Often the heaviest compute step
- ==Tiling==: Converting processed vector data into [[Tile]]s (via [[Tippecanoe]]) or [[Raster]] data into [[Cloud-Optimized GeoTIFF|COG]]s or [[Tile Pyramid]]s.


# Load: Where it ends up
- [[PostGIS]] for queryable vector storage
- Object storage ([[Amazon S3|S3]], [[Cloudflare R2|R2]]) for [[Cloud-Optimized GeoTIFF|COG]]s, [[FlatGeobuf]]s, [[PMTiles]]
- [[Tile Server]]s ([[Martin]], [[pg_tileserv]]) for serving [[Vector Tile]]s
- Data warehouses ([[BigQuery]], [[Snowflake]], [[DuckDB]]) for analytics
- Search indexes ([[ElasticSearch]] with geo support) for proximity search



___________

# Scale Challenges specific to Spatial ETL
- Spatial partitioning:
	- You can't just partition your dat by rowID, you need to partition by geography, so workers processing adjacent tiles don't constantly need data from eachother's partitions.
	- ==As a result, [[S2 Geometry|S2]] or [[H3]] cell IDs are common partitioning keys==
- Geometry serialization overhead:
	- Geometries are expensive to serialize/deserialize across worker boundaries.
	- [[GeoArrow]] helps here by keeping geometry in a column binary format that doesn't need repeated parsing.
- Predicate pushdown limitations:
	- In regular ETL, you can push filters down to the source, but spatial filters are harder:
		- You can push bbox filters to [[Cloud-Optimized GeoTIFF|COG]] and [[FlatGeobuf]]s, but ==complex polygon intersection require reading more data than you need and filtering in memory.==
- [[Antimeridian]] and pole edge cases:
	- Pipelines that work perfectly for continental US data break on data near 180 longitude or the poles. Geometries that cross the antimeridian need to be split before many spatial operations work correctly.


# Common Tools
- [[GDAL]]/ogr2ogr: The swiss army knife for handling format conversion
- [[Tippecanoe]]: [[Vector Tile]] generation from large vector datasets
- [[rasterio]] + [[Fiona]]: Python-native Raster and Vector processing
- [[Geopandas]]:  Good for moderate-scale vector ETL, hits memory limits on large dataset.
- [[Dask]] + Dask-GeoPandas: Parallelized GeoPandas for larger datasets.
- [[Apache Sedona]]: [[Apache Spark|Spark]]-based, for truly large-scale (billions of geometries)
- [[DuckDB]] Spatial: Increasingly popular for medium-scale analytical ETL, fast, no cluster needed


# Pipeline Example:
Say you wanted to build a dataset of every building in the US enriched with census tract attributes (NOTE: This example isn't true because Census data is at aggregated geographic levels, not individual households, but roll with it):
1. Extract: Download Overture Maps buildings ([[FlatGeobuf]], ~50GB)
2. Extract: Download Census TIGER tract boundaries ([[Shapefile]])
3. Reproject both to [[EPSG Code|EPSG]]:5070 (Albers equal area, good for CONUS)
4. Fix geometries (`ST_MakeValid` on both)
5. Spatial join: Join the buildings to their relevant census tracts (point-in-polygon operation at scale)
6. Aggregate: Count buildings per tract, compute median footprint area, etc.
7. Load: Upload results to PostGIS for querying, or GeoParquet for analytics
8. Tile: Building footprints -> [[PMTiles]] via [[Tippecanoe]] for visualizations.
	- PMTiles are most commonly seen serving basemaps, but that's not a format limitation! PMTiles can contain either [[Vector Tile]]s (e.g. [[Mapbox Vector Tile|MVT]]) or [[Raster Tile]]s.
	- The basemap association comes from the fact that Protomaps distributes a whole-world vector basemap as a single PMTiles file, which is many peoples' first encounter with the format, but the container itself is just "tiles at zoom/x/y coordinates", and it doesn't care what's inside each tile.
	- Building footprints as vector PMTiles makes sense! At low zoom levels, Tippecanoe drops or simplifies buildings, and at high zoom you get full footprint geometry. 










