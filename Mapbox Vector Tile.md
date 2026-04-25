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

[[PMTiles]] is a [[Cloud-Native Geospatial|Cloud-Optimized]] format for storing millions of [[Mapbox Vector Tile]] files in an efficient, accessible manner via HTTP range requests.

MBTile vs MVT (similar name):
- MVT is a format spec for a single tile; protobuf-encoded vector features for one Z/X/Y tile.
- MBTiles is a container format, a SQLite database that stores many tiles (raster or vector) in a single file.
- MBTiles is often used to store MVT tiles! Each row in the `tiles` table holds one gzipped MVT blob.




![[Pasted image 20260425104955.png]]

![[Pasted image 20260425105044.png]]
Note:  When you get `/tiles/14/8732/5354.pbf`, you receive a single [[Mapbox Vector Tile|MVT]] file representing exactly one tile (encoded as [[Protobuf]]), which contains many layers, each with many vector features.
- In other formats, sometimes tiles are bundled together, such as in [[MBTile]]s (SQLite database containing many files), or [[PMTiles]] (single cloud-optimized archive of many tiles)






