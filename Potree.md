The ==dominant open-source [[WebGL]]-based point cloud renderer== that lets you visualize massive [[Light Detection and Ranging|LiDAR]] datasets in the browser (billions of points, interactive).

A national LiDAR dataset might have 500 billion points; you can't load that into a browser, or even into memory on most machines. 
- Potree solves this with a streaming level-of-detail approach built on the [[Octree]] structure of [[Entwine Point Tiles]]:
	- Camera zooms out: Load onlly the root node (coarse, few points, whole dataset)
	- Camera zooms in: Progressively load deeper octree nodes (for visible region)
	- Full zoom: Load leaf nodes (full density), only for the small area in view

==At any moment, Potree only loads and renders the nodes relevant to your current viewpoint and zoom level, and the rest stays on the server.==
