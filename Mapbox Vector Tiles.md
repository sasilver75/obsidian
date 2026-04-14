---
aliases:
  - MVT
---
[[Mapbox Vector Tiles]] (MVT; .mvt/.pbf)
	- ==The dominant web vector tile format==, commonly served by [[Tile Server]]s.
	- The map is divided into a grid of tiles at each zoom level (standard {z}/{x}/{y} scheme).
	- Each tile contains the vector features (geometries, properties) clipped and simplified for that tile's bounding box and zoom level.
	- Encoded as [[Protobuf|Protocol Buffer]]s (binary).












