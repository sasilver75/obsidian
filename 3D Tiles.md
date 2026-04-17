Developed by [[Cesium]], 3D Tiles is now an open [[Open Geospatial Consortium|OGC]] `standard` (since 2019) ==for streaming massive heterogenous 3D geospatial datasets==:
- City-scale [[Photogrammetry]] meshes
- [[Light Detection and Ranging|LiDAR]] point clouds
- [[Building Information Modeling|BIM]] models
- Terrain

It's ==the 3D equivalent to [[Tile Pyramid]]s==, but for volumetric 3D content. Similar to [[Entwine Point Tiles|EPT]] for point clouds, using a hierarchical spatial tree.

Problem:
- A [[Photogrammetry|Photogrammetric]] mesh of a whole city might be 500GB of geometry and textures!
	- A national [[Light Detection and Ranging|LiDAR]] dataset might be trillions of points: You can't load these in the browser, or even into memory on most machines!
- You need a way to:
	- Show the whole dataset at once, at low detail
	- Progressively load more detail as the camera zooms in
	- Never load data that's not visible
	- Stream efficiently over [[HTTP]]
==This is what 3D tiles solves, in 3D space!==

A 3D Tiles dataset is called a ==tileset==, described by a `tileset.json` file:
```
  {
    "asset": {"version": "1.1"},
    "geometricError": 500,
    "root": {
      "boundingVolume": {"region": [-1.3, 0.6, -1.2, 0.7, 0, 500]},
      "geometricError": 200,
      "refine": "ADD",
      "content": {"uri": "root.glb"},
      "children": [
        {
          "boundingVolume": {...},
          "geometricError": 50,
          "content": {"uri": "tile_0_0.glb"},
          "children": [...]
        }
      ]
    }
  }
```

Each tile has:
- ==Bounding volume==: The spatial extent of the tile; three options:
	- region: lat/lng/height box (geographic)
	- box: oriented bounding box in 3D space
	- sphere: bounding sphere
- ==Geometric error==: A number in meters representing how much detail is lost by displaying this tile instead of its children.
	- The root tile of a city might have a geometric error of 500 (very coarse), while a leaf tile of a single building might have geometric error 0.1 (nearly perfect).
- ==Content==: The actual 3D data for this tile. In 3D Tiles 1.1, this is a [[GL Transmission Format]] (glTF) file.
- ==Children==: The more detailed tiles that replace or add to this tile when the camera gets closer.

## Refinement strategies (How children relate to parents is controlled by the "refine" property):
- REPLACE: Children completely replace the parent; used for photogrammetry meshes.
- ADD: Children *add* detail on top of the parent. Used for point clouds
![[Pasted image 20260417163426.png]]

## The Level of Detail selection algorithm
- The renderer decides which tiles to load and render using the ==Screen Space Error (SSE)==: Projecting the tile's geometric error into screen pixels:
```
  SSE = (geometricError × screenHeight) / (distance × 2 × tan(fov/2))
```
  - If SSE > threshold (typically 16 pixels): this tile is too coarse, load children
  - If SSE ≤ threshold: this tile is detailed enough, render it, stop subdividing
- ==The whole visible tileset is traversed every frame, making load/unload decisions dynamically as the camera moves.==


# Beyond CesiumJS
- Obviously ingested by [[Cesium]], but is an [[Open Geospatial Consortium|OGC]] standard, so other tools have adopted it:
	- [[deck.gl]] has a `Tile3DLayer` that renders 3D tiles, useful when you want 3D tiles in a deck.gl application, rather than a full Cesium viewer.
	- [[QGIS]]: Experimental 3D tile support
	- Unreal Engine
	- Unity
	- [[ArcGIS]]: Supports 3D tiles in Scene Viewer and ArcGIS Pro


3D Tiles is increasingly the foundation for digital twins — real-time 3D representations of physical infrastructure synchronized with sensor data and operational systems.




