---
aliases:
  - NetCDF
---
A ==binary file format for storing multidimensional array data==, dominantly used in climate science, oceanography, meteorology, and atmospheric research; the standard format in those fields.

NetCDF was designed for local disk access. On [[Blob Storage|Object Storage]] it ==has serious problems==:
  - A single file can be hundreds of GB
  - Reading a small spatial subset still requires many small HTTP range requests to
  traverse the chunked structure
  - The file must be fully written before it can be read (no streaming writes)
  - No support for parallel writes
  
  This is ==precisely why [[Zarr]] was created== — it takes the same conceptual model (labeled
  multidimensional arrays with chunks and metadata) but stores each chunk as a separate object in S3, making parallel reads, partial reads, and parallel writes natural. Zarr is increasingly replacing NetCDF for cloud-native workflows while NetCDF remains dominant in traditional HPC/supercomputer environments.

Built around labeled, multidimensional arrays. Rather than a table or an image, NetCDF naturally represents data with many dimensions:
```
  temperature[time, pressure_level, latitude, longitude]
  precipitation[time, latitude, longitude]
  wind_u[time, altitude, latitude, longitude]
  wind_v[time, altitude, latitude, longitude]
```
Each dimension has a name and coordinate values.

# Structure
A NetCDF file contains:

Dimensions — named axes with a size
```
  time: 365
  latitude: 180
  longitude: 360
  level: 37
```
  
  Variables — arrays defined over some subset of dimensions
  ```
  temperature(time, level, latitude, longitude)  → float32
  sea_surface_temp(time, latitude, longitude)    → float32
  latitude(latitude)                             → float64 (coordinate variable)
  ```

  Attributes — metadata attached to variables or the whole file
  ```
  temperature.units = "K"
  temperature.long_name = "Air Temperature"
  temperature.missing_value = -9999
  file.institution = "ECMWF"
  file.history = "Created 2023-01-15"
  ```




