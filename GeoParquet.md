References:
- [Cloud-Optimized Geospatial Formats Guide: GeoParquet](https://guide.cloudnativegeo.org/geoparquet/)

See also:
- [[Cloud-Optimized GeoTIFF]], an analogous cloud-optimized datatype instead for ==Raster== data.
	- While GeoParquet is great for things like building footprints, road networks, and POIs, COGs are great for elevation data, satellite imagery, and land cover classification.


An ==cloud-optimized file format for storing geospatial [[Vector]] data (==points, lines, polygons) in [[Apache Parquet|Parquet]], a popular ==columnar storage format== for tabular data. Adds how to encode geometries in the geometry column, and how to include metadata like the geometries' [[Coordinate Reference System]] and bounding box.
- GeoParquet stores geometries in standard [[Well-Known Binary]] (WKB), so it supports any vector geometry type defined in the [[Simple Features Access|OGC Simple Features]] specification.
- Any program that can read Parquet can read GEoParquet as well, even if it doesn't make sense of the geometry information.


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

Has a symbiotic relationship with the [[GeoArrow]] specification, which makes it easy to use spatial vector data *in memory.* You load a GeoParquet file into GeoArrow.


![[Pasted image 20260419122500.png]]
Above:
- A Parquet ==file== consists of a sequence of chunks called ==row groups==, which are logical groups of columns with the same number of rows.
- A row group consists of multiple columns, each called a ==column chunk==, which are sequences of raw column values that are guaranteed to be contiguous in the file.
- All row groups in the file must have the same schema, meaning that the data type of each column must be the same for every row group.
- A parquet file also includes ==metadata== describing the individual chunking, which includes hte byte range of every column chunk in the dataset, letting the Parquet reader fetch any given column chunk once they have the file metadata.
	- A Parquet metadata also includes *column statistics* (min/max value) for each column chunk, so if a user is interested in data where column "A > 100", the reader can skip loading/parsing any column chunks where the max value is known to be less than 100.
	- Interestingly, the metadata is located at the END of the file, which makes it easier to write (initially), as you don't need to know how many total rows you have at the beginning, but slightly harder to read (in practice, not too much more).
- Because Parquet is internally chunked, Parquet can fetch specific row groups that meet a filtering condition, and because it's column-oriented, only specific columns that the user is interested.


- Parquet has internal chunking; instead of storing all data for a certain column in a single contiguous region, it stores all of the data for the first 100,000 rows in one big block, and the next 100,000 rows for the column in the next block, etc.
- All of this is contained in the metadata at the top of file; readers can take advantage of range requests to just get what they want.
- Above (Confirm): It sees you can store multiple columns in same chunk?


![[Pasted image 20260416154527.png]]
- The spatial partitioning in GeoParquet lets you choose how you sort the data.
- The Overture Maps dataset is one that has data in GeoParquet... 

![[Pasted image 20260419015943.png]]
Source: https://guide.cloudnativegeo.org/overview.html#/geoparquet 



![[Pasted image 20260425102801.png|1367]]