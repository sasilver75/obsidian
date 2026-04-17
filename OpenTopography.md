An [[National Science Foundation|NSF]]-funded platform (hosted at UCSC) that provides free, easy access to high-resolution topographic data, primarily [[Light Detection and Ranging|LiDAR]], along with on-demand processing tools.

It's essentially a ==data portal and processing service for elevation/topographic data==.
- Before Open Topography, accessing LiDAR was genuinely painful.
- ==OpenTopography is the first stop for grabbing a DEM for a study area quickly or checking what LiDAR coverage exists for a region==
	- For serious LiDAr work, you'd eventually move to raw LAZ files from USGS 3DEP or a state program, process with PDAL or lidR, and build your own derived products, but OpenTopography dramatically lowers the barrier to getting started.

Notably provides:
- US National LiDAR data from [[United States Geological Survey|USGS]]'s [[3DEP]].
- Hundreds of community-contributed regional datasets
- Global [[Digital Elevation Model]]s: [[Shuttle Radar Topography Mission|SRTM]], [[Copernicus DEM|Copernicus]] [[Copernicus DEM|GLO-30]], [[ALOS AW3D30]], [[NASADEM]], all through a single API
- Some international LiDAr datasets

On-demand processing:
- Rather than downloading raw point clouds and processing them yourself, OpenTopography lets you draw a bounding box and request derived products.

Limitations:
- On-demand processing has file size limits for free users
- Not truly cloud-native: You download files rather than doing range-request acess in place.