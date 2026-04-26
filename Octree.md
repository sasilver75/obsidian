
A 3D [[Geospatial Index|Spatial Index]] that recursively subdivides a volume (3D space) into 8 equal sub-cubes (hence "oct"), each of which can be further subdivided.
- Used by [[Cloud-Optimized Point Cloud|COPC]], a cloud-optimized point cloud data format; [[Light Detection and Ranging|LiDAR]] scenes have billions of points, and an Octree lets you do:
	- Spatial queries: "Give me all points within this bounding box", traverse sonly relevant branches, not all points.
	- Level-of-detail: Each Octree level is a coarser representation. Zoom out -_> sample one point per node, Zoom in -> descend to finer nodes.
	- Out-of-core processing: When the data is too large to fit in ram, so you have to process it in chunks from disk, nodes map to contiguous chunks on disk, so you can only load what's needed.
- It's the 3D analog of a [[QuadTree]] (which splits *2D space* into *4 squares*).

1. Start with a bounding cube covering your entire dataset
2. Split it into 8 equal children
3. Each child that contains enough points gets split again
4. Repeat until leaves contain few enough points (or you hit max depth)

  Root cube
  ├── Child 1 (NW top)     → split again if dense
  ├── Child 2 (NE top)     → leaf (sparse enough)
  ├── Child 3 (SW top)     → split again
   │   ├── ...8 children
  └── ... 8 total children







![[Pasted image 20260425235853.png]]