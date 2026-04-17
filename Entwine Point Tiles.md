---
aliases:
  - EPT
  - Entwine
---
A ==cloud-friendly format for storing and serving large LiDAR point clouds==, developed by [[Hobu Inc.]], the same people behind [[Point Data Abstraction Library|PDAL]].
- ==Analogous to [[Zarr]]== but for point cloud data; solves the same problems for [[LASer|LAS]] and [[LAZ]] that [[Cloud-Optimized GeoTIFF|COG]]s and [[Zarr]] solve for [[GeoTIFF]]. 
	- The current standard for national-scale distribution.
- A national LiDAR dataset might have hundreds of billions of points; without a hierarchical structure, loading it for visualization or spatial queries requires reading everything; EPT lets you:
	- Stream point clouds in web viewers (e.g. [[Potree]], [[Cesium]])
	- Clip to a bounding box efficiently (only read nodes that intersect your region)
	- Progressive loading (show coarse overview first, refine as you zoom in)

Compare with: [[Cloud-Optimized Point Cloud]] (COPC), which uses a philosophy to [[Cloud-Optimized GeoTIFF|COG]], but for point clouds.

==Entwine== is an open source tool (by Hobu, the [[Point Data Abstraction Library|PDAL]] people) that converts [[LAZ]] files into [[Entwine Point Tiles|EPT]] format for cloud-native serving.
- Reads your LAZ tiles, build the octree, writes EPT to local disk or S3.

It organizes a point cloud as a hierarchical octree (a 3D analogue to [[QuadTree]]s that recursively subdivides spaces), and uses a multi-file format like Zarr:
```
  ept.json                    (metadata: bounds, CRS, point count, schema)
  ept-data/
    0-0-0-0.laz              (root node: all points, heavily subsampled)
    1-0-0-0.laz              (octree level 1, octant 0)
    1-1-0-0.laz
    1-0-1-0.laz
    ... (8 children per node)
    2-0-0-0.laz              (octree level 2, finer detail)
    ...
    14-3821-6724-2.laz       (leaf nodes: full density data)
  ept-hierarchy/
    0-0-0-0.json             (which nodes exist, point counts per node)
```
- ==Each file is a [[LAZ]] file== containing the points that fall in that octree node at that level of detail.
	- ==Higher levels (closer to root) contain subsampled points representing the whole dataset at coarse resolution. Lower levels contain progressively denser subsets of smaller spatial regions.==

# How Access Works
- To get the whole dataset at a low level of detail, just read the root node file.
- To zoom into a region, read the nodes covering that region at appropriate depth.
- For full density into small area, read the leaf node for that spatial extent.

(This is the ==3D equivalent of a [[Cloud-Optimized GeoTIFF|COG]]'s overview pyramid; you get appropriate detail for your current view without reading the whole dataset.==)


![[Pasted image 20260417145805.png]]
