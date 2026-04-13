Zarr is a geospatial data format (c.f. [[Cloud-Optimized GeoTIFF|COG]]s).

Instead of storing a large array as one monolithic file, Zarr breaks it into small chunks, independently compressed binary files.
- So a 10GB climate dataset might be stored as thousands of small `.zarr` chunk files in a directory or object store.

For geospatial, [[Raster]] data is often multi-dimensional ({lat, lon, time, variable}, e.g. temperature at e very grid cell for every hour for 40 years). 
- Traditional data formats require you to download the whole file before reading any of it.
- Zarr lets you say "give me just the Pacific Northwest, for the month of July 2023", and only the relevant chunks are fetched.

HTTP Range requests let you fetch a specific byte range from a remote file.
- [[Cloud-Optimized GeoTIFF]] (COG) uses this for a 2D [[Raster]] array.
- Zarr extends the same idea to N-dimensional arrays
	- Each chunk is a separate object in [[Amazon S3]], so you can fetch exactly what you need in parallel.

Zarr is almost always used with the `Xarray` library, which is a Python library for labeled N-dimensional arrays (Pandas-like, it sounds like). 
- Xarray uses Zarr as a storage backend the same way that Pandas uses [[Apache Parquet|Parquet]].


