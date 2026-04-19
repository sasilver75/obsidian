---
aliases:
  - CNG
  - Cloud-Native Geospatial Foundation
  - Cloud-Optimized
---
References:
- [Cloud-Native Geospatial Data Format Glossay](https://guide.cloudnativegeo.org/glossary.html)

A philosophy, set of formats, and architectural patterns for storing, accessing, and processing geospatial data in a way that's designed from the ground up for cloud [[Blob Storage|Object Storage]], rather than adapting traditional GIS workflows to the cloud.

The shift to cloud-native geospatial is analogous to what happened to web applications moving from server-rendered pages to APIs + client-side rendering; it fundamentally changes the architecture.
- Analysis that required downloading TBs can now run against data in place
- Global datasets are freely accessible without data transfer via AWS/GCP open data programs
- Serverless architectures replace tile servers and GIS servers
- The barrier to planetary-scale analysis drops dramatically

The [[Cloud-Native Geospatial|Cloud-Native Geospatial Foundation]] (CNG) is a [[Linux Foundation]] project that stewards the key open standards and promotes their adoption. It oversees:
- [[SpatioTemporal Asset Catalog|STAC]] specification
- [[Cloud-Optimized GeoTIFF|COG]] specification
- [[GeoParquet]] specification
- [[Cloud-Optimized Point Cloud|COPC]] specification
- [[PMTiles]]
The foundation brings together organizations (AWS, Microsoft, Google [[ESRI]], [[Planet Labs|Planet]], [[Maxar]], and others) to coordinate development of these open standards and avoid fragmentation.

# Why Cloud Optimization?
- Geospatial data is experiencing exponential growth in both size and complexity; traditional data access methods (file downloads) have become increasingly impractical for achieving scientific objectives. 
- Cloud optimization of file formats enables efficient, on-the-fly access to geospatial data, offering:
	- Reduced latency (Subsets of the raw data can be fetched and processed much faster than downloading files)
	- Scalability (stored on cloud object storage, which is infinitely scalable)
	- Flexibility (high-levels of customization, advanced query capabilities without downloading)
	- Cost-effectiveness (reduced data transfer and storage needs (compression))

# Core Principles
- ==Data stays where it is==
	- Data formats are designed so that compute comes to the data, not the other way around.
	- [[HTTP Range Request]]s let you read specific byte ranges of a file on [[Amazon S3|S3]] without downloading the whole thing. A [[Cloud-Optimized GeoTIFF|COG]] on S3 serves exactly the tiles and overview levels you need for a given query, and nothing more.
- ==Formats optimized for partial access==
	- Every cloud-native geospatial format shares a design pattern: an indx or header structure at some known location, followed by data organized so that spatial subsets can be retrieved with a small number of [[HTTP Range Request]]s:
		- [[Cloud-Optimized GeoTIFF|COG]]: Internal tile index + overview pyramid
		- [[FlatGeobuf]]: Packed Hiltbert-sorted [[R-Tree]] index + feature data
		- [[GeoParquet]]: Parquet row group statistics for spatial [[Predicate Pushdown]]
		- [[Cloud-Optimized Point Cloud|COPC]]: Octree index for point cloud spatial subsetting
		- [[Zarr]]: Chunk-per-object, each chunk independently retrievable
		- [[PMTiles]]: single-file tile archive with spatial index
- ==Serverless where possible==
	- A [[DuckDB]] query against a [[GeoParquet]] file on [[Amazon S3|S3]], a [[MapLibre]] map reading [[PMTiles]] from [[Cloudflare R2|R2]], a [[rasterio]] script reading a [[Cloud-Optimized GeoTIFF|COG]]... none of these require running a [[Tile Server]], database server, or processing server; just object storage and client-side computation.
- ==Separation of storage and computer==
	- Data lives in [[Blob Storage|Object Storage]] ([[Amazon S3|S3]], [[Google Cloud Storage|GCS]], [[Cloudflare R2|R2]]) independently on any particular compute environment.
	- The same COG can be read by a Python script, a web browser via [[TiTiler]], a [[QGIS]] user, or a [[Apache Spark|Spark]] cluster, without moving or copying the data. 
	- Storage is cheap and durable; compute is ephemeral and elastic.


## Cloud-Optimized Format Pattern
1. Metadata or specification provides addresses for data blocks.
2. Metadata is stored in a consistent format and location.
3. All metadata can be loaded by a single read operation.
4. Metadata or the format specification can be used by libraries to read subsets of data from the underlying file format. Subsetted access is facilitated via addressable chunks, internal tiling, or both.
==The above characteristics allow for both parallelized and partial reading.==

Questions to ask when generating Cloud-Optimized Geospatial Data in any format:
1. What variable(s) should be included in the new data format?
2. Will you create copies to optimize for different needs?
3. What is the intended use case or usage profile? Is this for visualization, analysis, or both?
4. What is the expected access method?
5. How much of your data is typically rendered or selected at once?

# The Stack
- [[Vector]] Data
	- [[GeoParquet]] for analytics
	- [[FlatGeobuf]] for streaming, spatial queries
	- [[PMTiles]] for map display
- [[Raster]] Data
	- [[Cloud-Optimized GeoTIFF|COG]] for imagery, single scenes
	- [[Zarr]] for multidimensional arrays, time series, large data cubes
- Point Clouds
	- [[Cloud-Optimized Point Cloud|COPC]]: Single-file cloud-native [[Light Detection and Ranging|LiDAR]]
	- [[Entwine Point Tiles|EPT]]: Tiled, cloud-native LiDAR
- Catalogs/Discovery
	- [[SpatioTemporal Asset Catalog|STAC]]: Standardized metadata and discovery
- Access Mechanism
	- [[HTTP Range Request]]: The underlying primitive that everything builds on
- Processing
	- [[rasterio]], [[Geospatial Data Abstraction Library|GDAL]]: [[Cloud-Optimized GeoTIFF|COG]] access
	- [[DuckDB]] Spatial: [[GeoParquet]] queries
	- [[Xarray]] + [[Dask]]: [[Zarr]]/[[Cloud-Optimized GeoTIFF|COG]] time series
	- [[Point Data Abstraction Library|PDAL]]: [[Cloud-Optimized Point Cloud|COPC]]/[[Entwine Point Tiles|EPT]] point clouds


![[Pasted image 20260419014455.png|700]]





