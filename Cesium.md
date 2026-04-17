---
aliases:
  - CesiumJS
---
A [[WebGL]]-based tool for 3D globe rendering, used for 3D globe and map visualizations in the browser (Cesium is the company, Cesium JS is the frontend rendering engine).
- Used for flight paths, satellite imagery, and terrain.
- Focused on 3D, rather than 2D map views.

Unlike [[MapLibre GL JS]] or [[Leaflet]] which are fundamentally 2D (flat map with optional 3D extrusions), Cesium is genuinely 3D from the ground up.
- Earth is sphere
- Has real elevation
- Objects exist in 3D space
- You can fly through cities, go underground, view from space, tilt to ground level

What it can render:
- Terrain (Has its own terrain format, Quantized Mesh, and streams terrain tiles from Cesium Ion or self-hosted sources.)
- Imagery (Standard XYZ tile layers draped over the 3D terrain; supports multiple imagery providers simultaneously)
- 3D Tiles (Cesium invented the [[3D Tiles]] specification (now an [[Open Geospatial Consortium|OGC]] standard), which is a spatial data format for streaming massive 3D datasets)
	- Photogrammetry meshes (textured city model from drone/aerial imagery)
	- Point clouds (LiDAR)
	- Building models ([[Building Information Modeling|BIM]] data, [[CityGML]])
	- Vector data in 3D


==CZML== is Cesium's own JSON format for describing time-dynamic scenes (things that move over time): satellites moving along orbits, vehicles following paths, sensors sweeping areas.

Entities and primitives: Points, polylines, polygons, labels, billboards (screen-facing images), models (glTF), all placeable anywhere on the globe with full 3D positioning.

You can place arbitrary 3D models ([[GL Transmission Format|glTF]] models; buildings, vehicles, sensors, satellites) at real-world coordinates.

==Cesium Ion== is their commercial cloud platform that they operate alongside the open-source library, providing:
- Hosted terrain ("Cesium World Terrain" is global high-resolution)
- Hosted imagery (Bing Maps, [[Sentinel|Sentinel-2]] global mosaic)
- Asset hosting and tiling: Upload a point cloud, photogrammetric mesh, or 3D model and Ion tiles it into 3D tiles and serves it.
- The tiling pipeline that converts raw data ([[LASer|LAS]], OBJ, DAE, [[GeoTIFF]]) into streamable 3D tiles.

You can use CesiumJS completely open source without Ion, but then you need to self-host terrain and tile your own data.




