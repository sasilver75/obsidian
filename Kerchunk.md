
A Python ==library for creating reference files== to support cloud-optimized access to traditional geospatial file formats (like [[Network Common Data Form|NetCDF]] or [[Hierarchical Data Format 5|HDF5]]). Kerchunk ==negates the need to create and store copies of data for cloud-optimized access==.

==Kerchunk reference files are [[JSON]] files with Key-Value pairs for reading the underlying data as a [[Zarr]] data store==. The keys are Zarr metadata paths or paths to Zarr data chunks, and the values will either be raw data values or a list of the file URL, starting byte, and byte length where the data can be read.

Given that Kerchunk creates [[Zarr]] metadata for non-Zarr data, Kerchunk is compatible with Zarr tools that can use Zarr.

==Kerchunk enables a unified way to access chunked, compressed, n-dimensional data across a variety of conventional data formats== ([[Network Common Data Form|NetCDF]]/[[Hierarchical Data Format 5|HDF5]]/[[GRIB2]]/[[Tagged Image File Format|TIFF]]).
- "*Kerchunk provides a method of providing cloud-optimized access to data that is in more traditional archival formats.*"



```python
import fsspec
import json
from kerchunk.hdf import SingleHdf5ToZarr

local_file = 'some_data.nc'
out_file = 'some_references.json'

# Instantiate the local file system with fsspec to save kerchunk's reference data as json.
fs = fsspec.filesystem('')
in_file = fs.open(local_file)

# The inline threshold adjusts the size below which binary blocks are included directly in the output.
# A higher inline threshold can result in a larger json file but faster loading time overally, since fewer requests are made.
h5chunks = SingleHdf5ToZarr(in_file, local_file, inline_threshold=300)
with fs.open(out_file, 'wb') as f:
    f.write(json.dumps(h5chunks.translate()).encode())
```
Note: The `fsspec` library provides a uniform file system interface for many different storage backends and protocols.
