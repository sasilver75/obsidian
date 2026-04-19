---
aliases:
  - COPC
---
References:
- [Cloud-Optimized Geospatial Formats Guide: COPC](https://guide.cloudnativegeo.org/copc/) 
- [COPC site](https://copc.io/)

See also: [[Entwine Point Tiles]]


In the context of [[Point Cloud]]s (e.g. [[Light Detection and Ranging|LiDAR]]), the LASER ([[LASer|LAS]]) file format is designed to store 3-dimensional (x,y,z) point cloud data typically collected from LiDAR. [[LAZ]] files are compressed LAS files, and ==[[Cloud-Optimized Point Cloud|COPC]] files are cloud-optimized versions of LAZ files==.
- ==One LAZ File = one COPC file==; it reorganizes the points from the point cloud dataset into a spatially-indexed file use spatial octrees, ==allowing for partial reads==.
- For multi-file datasets (e.g. a LiDAR survey across across hundreds of LAZ files), you'd use a multi-file format like [[Entwine Point Tiles|Entwine]]. In some sense, [[Cloud-Optimized Point Cloud|COPC]]:[[Entwine Point Tiles|EPT]]::[[Cloud-Optimized GeoTIFF|COG]]:[[Zarr]], or you can think of 

[[Cloud-Optimized Point Cloud|COPC]] files play a similar role to what [[Cloud-Optimized GeoTIFF|COG]]s play for [[GeoTIFF]]s: Both are valid versions of the original file format, but with additional requirements to support cloud-optimized data access.
- In the case of COGs, there are additional requirements for tiling and overviews
- In COPC, data must be organized into a clustered octree with a variable-length record (VLR) describing the octree structure.






![[Pasted image 20260419015847.png]]


![[Pasted image 20260417145827.png]]