An open-standard spatial extension to the Apache [[Apache Parquet|Parquet]] file format, adding geospatial data types line points/lines/polygons.
- A file format for Geospatial [[Vector]] data.
	- (Compare with: [[Cloud-Optimized GeoTIFF]]s, which are for [[Raster]] data)
		- GeoParquet is great for things like building footprints, road networks, POIs.
		- COGs are great for remote sensing and continuous spatial phenomena (elevation data, satellite imagery, land cover classification

The geometry is stored as [[Well-Known Binary|WKB]] bytes in a column, with metadata describing both the [[Coordinate Reference System|Coordinate Reference System]] and bounding box.

Advantage:
- Get Parquet's columnar analytics performance on geospatial data.
	- [[DuckDB]] with its spatial extension can query a 10GB GeoParquet file with spatial filters in seconds without loading it into memory.

Drawbacks:
- No real concept of overviews; if you want to look at it while zoomed very far out, you might need to download 100MB to the browser. In a lot of real product contexts, that's not acceptable to the people that control the UX. This is the motivation fr a lot of tiled web maps.

Often seen for large open geospatial datasets distributed for analytics:
- Overture Maps
- Microsoft Building Footprints
- Google Open Buildings
... and any dataset where you want analytics-first access rather than map rendering.


Has a symbiotic relationship with the [[GeoArrow]] specification, which makes it easy to use spatial vector dat *in memory.* You load a GeoParquet file into GeoArrow.






__________________

![[Pasted image 20260416154322.png]]
- Parquet has internal chunking; instead of storing all data for a certain column in a single contiguous region, it stores all of the data for the first 100,000 rows in one big block, and the next 100,000 rows for the column in the next block, etc.
- All of this is contained in the metadata at the top of file; readers can take advantage of range requests to just get what they want.
- Above (Confirm): It sees you can store multiple columns in same chunk?


![[Pasted image 20260416154527.png]]
- The spatial partitioning in GeoParquet lets you choose how you sort the data.
- The Overture Maps dataset is one that has data in GeoParquet... 

