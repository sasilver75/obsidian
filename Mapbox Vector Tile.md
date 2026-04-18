---
aliases:
  - MVT
---
[[Mapbox Vector Tile]] (MVT; .mvt/.pbf)
	- ==The dominant web [[Vector Tile]] format==, commonly served by [[Tile Server]]s.
		- The actual binary time format inside [[PMTiles]]/[[MBTile]]s.
	- The map is divided into a grid of tiles at each zoom level (standard {z}/{x}/{y} scheme).
	- Each tile contains the vector features (geometries, properties) clipped and simplified for that tile's bounding box and zoom level.
	- ==Encoded as [[Protobuf|Protocol Buffer]]s (binary).==

It's an open standard now maintained by the [[Open Geospatial Consortium]] (OGC), widely used beyond Mapbox.

Compare with [[OpenMapTiles]]









