The compressed version of [[LASer|LAS]], always used when the data is stored.
- The cloud-native successor is [[Cloud-Optimized Point Cloud]] (COPC), which is [[LAZ]] organized into a spatial octree so you can use [[HTTP Range Request]]s to just retrieve the points covering a specific area/resolution without downloading the whole file. It's the same data as LAZ, just different internal organization.
- *"[[LASer|LAS]] is the point cloud version of a [[Comma-Separated Values|CSV]]; every row is a point with coordinates and attributes. [[LAZ]] is that CSV gzipped, and [[Cloud-Optimized Point Cloud|COPC]] is that CSV spatially indexed for cloud access"* 

