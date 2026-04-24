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





__________

In 2010 Google came out with [[Google Earth Engine|GEE]]; they put a bunch of data in it, and you can operate on the data at a planetary scale. It's still a service that Google provides, and really showed what could be done with processing of big data in the cloud. It has a few different components:
- Unified Data Discovery: Search for data across multiple different datasets
- Data Orchestration Pipeline to ingest data and process it, somewhere in there.
- Scalable file access: We don't quite know how this works
- Analytics scalable to the planetary scale
- You can visualize it.

![[Pasted image 20260423143426.png]]
This is basically what we've reinvested in Cloud-Native Geospatial!

But GEE is not:
- Open source
- Based on standards
- Deployable 
- Easy to integrate with commercial data providers
- Easy to build things on top of it


So CNG: ==A collection of APIs and technologies for programmatic discovery, access, and exploitation of geospatial data (in the cloud).==
- Enter: Standard and Open Source.
- Replicate, in the open source world, each of the components mentioned above


Unified Data Discovery:
- [[SpatioTemporal Asset Catalog|STAC]] rose up, which is based on [[GeoJSON]], as well as [[OGC API Features]], which together represent APIs for serving up [[Raster]] and [[Vector]] data.
	- Q: Should I use [[SpatioTemporal Asset Catalog|STAC]] or [[OGC API Features|API Features]]?

STAC Items: GeoJSON, defining the region where there is data, and may contain additional properties.
- The data is in separate assets files.

OGC Features: Also GeoJSON, defining regions uniform properties, but there are no actual separate assets. 

So you should ask yourself: Does my data fit in STAC?
- If my geometry is the data itself, then use [[OGC API Features]].
- If my  geometry is really not the data, and it's just metadata about where data i located, use [[SpatioTemporal Asset Catalog|STAC]].

Things like labels for ML... don't use STAC.


For Data Orchestation
![[Pasted image 20260423165956.png]]
There isn't a single solution for any of this: At E84 we do a lot of work for small satellite companies, etc.
- There are a lot of solutions out there: [[Apache Airflow|Airflow]], [[Prefect]], and other generalized orchestration platforms.
- Planetary Computer uses something called pctasks for ingestion, but it's really specific to planetary computer (though it's open-source)
- At [[Element84 Earth Search|Element84]], they use something called Cirrus which is open-source, completely based on AWS, using [[AWS Step Functions]], [[DynamoDB]], etc... it's what we use for indexing geospatial data on AWS.
- [[STAC Workflow Open Orchestration Platform]] (SWOOP) is something they're working on , which is an orchestration platform they're working on... will expose an API using [[OGC API Processes]].

![[Pasted image 20260423170300.png]]
Instead of file-in, file-out, and using file names to embed metadata about your files...
Instead write workflows taking in metadata (perhaps you modify the metadata, generating new STAC items, generating new assets)... if you generate new assets, put them in the cloud, and put the location of those things in the metadata that you return!
- In the STAC metadata, can put a "derived from" field that says that a certain item is derived from another item, and put in a link. You can put in versioning information, etc:

![[Pasted image 20260423170432.png]]
So metadata can become your atomic unit of processing, rather than the data files themselves.


Now scaling file access:
![[Pasted image 20260423170448.png]]

[[Cloud-Optimized GeoTIFF]]: Raster (good for EO)
- Has an IFD (Image File Directory) at beginning of file, educing data seek times
	- Think: A table of contents. You can access this an d tell where all the different pieces/chunks/tiles are in your file
- Uses Tiled Segments; Decompress nad load data separately.
- Overviews in file: Efficiently use the data at any resolution
![[Pasted image 20260423170806.png]]
Standard [[Tagged Image File Format|TIFF]] file; things are laid out in striped, and the stripes are compressed individually; you have to read the entire thing, decompress it, and pull out the pixels you want.

The better way is using Tiles!
![[Pasted image 20260423170841.png]]
This is more closely aligned with the actual areas of interest that you have.


So ==a cloud-native dataset== is really one with ==small, addressable chunks==, via ==files, internal tiles, or some combination of the two.==


[[Zarr]]: Raster (good for modeled data)
- Works in a similar way! Chunked, compressed, N-dimensional arrays
- Similar to how [[Hierarchical Data Format 5|HDF5]] stores data
- It's an ==Exploded File Format==; rather than store all of the tiles in a single file, all of the tiles are actually individual files on Blob storage, so a Zarr dataset, rather than being one file... it's a bunch of files! You can read it very reasily in a similar fashion with multiple workers...
	- (2026) Drawback: Accessing each blob in Cloud storage is an API call, so it becomes very difficult to move or delete that data without making a ton of API calls, so one should think twice before really adopting Zarr in that way.
		- There's a change possibly in the works to combine these tiles backi into a file, which kind of brings us more back to the HDF5 model... so... we'll see.
- There's a [[Zarr|GeoZarr]] effort as well... the issue with Zarr...
	- Zarr came out of climatology work, where climatologists are dealing with homogeneous, global model data, where you have lots of variables.
	- In those cases, you have these lat/long arrays... you don't have something like you have... in the GDAL model where you have an affine transformation and projection. In Zarr you just have a lat/long array. And so... there lcearly was a need for having some sort of geospatial awareness within Zarr, and so the GeoZarr effort took off, which took off using CF (climate forecast) conventions...
	- This has been... there recently (2023) has been a working group for driving forward this.
	- In the future, we hoope that GeoZarr will enable something more like the GDAL data model for data that has been projected.



[[Cloud-Optimized Point Cloud|COPC]]: Point cloud
- A [[LAZ]] 1.4 file.... with metadata.... clustered according to that metadata
- There's a point-cloud feature in [[QGIS]] now (2023), and so that's been a great way to visualize point clouds.


[[GeoParquet]]: Vector
- Parquet for Geospatial
- The hope (2023) is that GeoParquet isn't another standard, it's just a way that geospatial is encoded in Parquet; it's just another datatype. Internally, the standard is quite simple, and it's currently (2023) at a v1.0 version.


There are also Cloud-optimized Tarballs, and [[Kerchunk]], which lets you create [[Cloud-Native Geospatial|Cloud-Optimized]] files out of really any legacy format, because really all you need is that table of contents and then chunks of data.


For scalable analytics: The Pangeo effort came around...
![[Pasted image 20260423172150.png]]



![[Pasted image 20260423172516.png]]




