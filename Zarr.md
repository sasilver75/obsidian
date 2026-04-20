---
aliases:
  - GeoZarr
---


References:
- [Cloud-Optimized Geospatial Formats Guide: Zarr](https://guide.cloudnativegeo.org/overview.html#/what-is-zarr)

> Zarr is a format for storing large-scale, compressed  n-dimensional or [[Raster]] data that's too big for users' machines; Zarr makes data small and organizes it in a way where users can take just the chunks they need, or distribute the load of processing these chunks across many machines.
> A Zarr chunk is an equally-sized block of the array in a larger Zarr store. Zarr chunks are normally stored in separate objects in object storage to make reading and updating individual chunks more efficient.
> Zarr is NOT designed for [[Vector]], [[Point Cloud]], or sparse data.


Zarr is a ==chunk-based, compressed, N-dimensional array format designed from the ground up for cloud [[Blob Storage|Object Storage]]==, taking the conceptual model of [[Network Common Data Form|NetCDF]]/[[Hierarchical Data Format 5|HDF5]] (labeled multidimensional arrays with metadata) and rebuilding it so that each chunk is a separate file/object rather than a byte range inside a monolithic file.
- This is a tradeoff:
	- [[Cloud-Optimized GeoTIFF|COG]] keeps everything in one file and uses HTTP range requests to fetch specific byte ranges.
		- Single object to manage/copy/move/delete, one URL to share, no per-chunk storage overhead.
		- "HTTP range requests are good enough, keep it simple"
		- For raster imagery; single 2D scenes or small temporal stacks. The canonical use-case is satellite imagery; one [[Landsat]] scene, [[Sentinel|Sentinel-2]] tile, or one [[Digital Elevation Model|DEM]]. A bounded spatial extent, a modest number of bands, and access patterns that are overwhelmingly "give me this spatial window at this resolution."
	- [[Zarr]] is genuinely multi-file, with a large Zarr store potentially having millions of objets in S3.
		- S3 LIST operations are slow and expensive, Per-request costs add up (millions of tiny chunks mean millions of API calls), eventual consistency complexity if you're writing many chunks in parallel.
		- "Parallel object storage is the future, optimize for that"
		- For large multidimensional arrays (weather model output, oceanographic data, datasets with many timesteps accessed in complex patterns) and data cubes spanning decades. The chunk-per-object model lets you read exactly the slice you need across an arbitrarily large cube.

Where they overlap:
- Large stacks of satellite imagery, say all [[Landsat]] observations over a region from 1984 to present, can be represented either way:
	- [[Cloud-Optimized GeoTIFF|COG]] per scene + [[SpatioTemporal Asset Catalog|STAC]] catalogue: The standard approach; Each scene is its own COG, and STAC tells you which scenes cover your area and time range: you load and stack them yourself.
	- [[Zarr]] datacube: All scenes pre-packed into one Zarr store, already aligned to a common grid. Faster to query, but expensive to build and maintain, and loses per-scene provenance.

![[Pasted image 20260417122647.png]]
Above: Zarr vs COG tradeoffs

Design Shift:
- HDF5/NEtCDF store chunks as byte ranges inside one big file (eg 10GB file):
```
chunk 0: bytes 0-4096
chunk 1: bytes 4096-8192
chunk 2: bytes 8192-12288
```
- Zarr stores data with each chunk as a separate object in Object Storage (eg [[Amazon S3|S3]]):
```
mydata.zarr/
    .zmetadata          (consolidated metadata)
    temperature/
      .zarray           (array metadata: shape, chunks, dtype, compressor)
      .zattrs           (attributes: units, long_name, etc.)
      0.0.0             (chunk at time=0, lat=0, lon=0)
      0.0.1             (chunk at time=0, lat=0, lon=1)
      0.1.0             (chunk at time=0, lat=1, lon=0)
      1.0.0             (chunk at time=1, lat=0, lon=0)
      ...
```
- Here, reading chunk 0.0.0 is one HTTP GET request, and reading 10 chunks is 10 independent GET requests that can happen in parallel.
	- Allows for parallel reads, parallel writes, and no overhead from traversal metadata (e.g. an internal [[B-Tree]] file header to find chunk locations); Instead, Zarr chunk locations are implicit from their filename (Chunk 2.3.1 is always at path 2/3/1; it takes one metadata read, then direct object access).


Instead of storing a large array as one monolithic file, Zarr breaks it into small chunks, independently compressed binary files.
- So a 10GB climate dataset might be stored as thousands of small `.zarr` chunk files in a directory or object store.

For geospatial, [[Raster]] data is often multi-dimensional ({lat, lon, time, variable}, e.g. temperature at e very grid cell for every hour for 40 years). 
- Traditional data formats require you to download the whole file before reading any of it.
- Zarr lets you say "give me just the Pacific Northwest, for the month of July 2023", and only the relevant chunks are fetched.

HTTP Range requests let you fetch a specific byte range from a remote file.
- [[Cloud-Optimized GeoTIFF]] (COG) uses this for a 2D [[Raster]] array.
- Zarr extends the same idea to N-dimensional arrays
	- Each chunk is a separate object in [[Amazon S3]], so you can fetch exactly what you need in parallel.

Zarr is almost always used with the [[Xarray]] library, which is a Python library for labeled N-dimensional arrays (Pandas-like, it sounds like). 
- Xarray uses Zarr as a storage backend the same way that Pandas uses [[Apache Parquet|Parquet]].

![[Pasted image 20260419015559.png]]

![[Pasted image 20260419015616.png]]

![[Pasted image 20260419015645.png]]

![[Pasted image 20260419015658.png]]




