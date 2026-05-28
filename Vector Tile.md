


A [[Mapbox Vector Tile]] contains the raw geometry coordinates, clipped to that tile's geographic boundary and quantized to a coordinate grid (typically ==4096x4096== units). 
- "Line from point (200, 300) to point (400, 350)"

How zooming works:
- Vector tiles are typically organized into a Z/X/Y [[Tile Pyramid]] like raster tiles... you have tiles are zoom 0, 1, 2, ... etc., but ==instead of each zoom level being a different resolution image== (like for [[Raster Tile]]s), ==each zoom level is a different level of geographic detail.==

Zoom 0: Simplified content outlines
Zoom 10: Tile contains streets, buildings at full detail
Zoom 14: Tile contains individual building footprints, addresses

This simplification is done at tile generation time by tools like [[Tippecanoe]].

When you zoom "between" zoom levels (e.g. zoom 10.6), frontend renderers like [[MapLibre GL JS]] just scale the rendered geometry, taking the zoom 10 tile and rendering it slightly larger. Because it's drawing vectors via [[WebGL]], not stretching pixels, it stays crisp, and then when you finally cross a zoom level threshold, it swaps in the next time with more detail, ideally seamlessly.


## Contents
- A ==single tile typically contains one or more **layers**== (e.g. "roads", "buildings", "water"), and each layer contains ==features== (eg a single building per feature).
	- Each feature has:
		- A Geometry expressed as integer coordinates on the tile's' 4096x4096 grid
		- Attributes/properties: Key-value pairs like {"name": "I-405", ...}

## Coordinate system
- ==Geometries are stored in tile-local integer coordinates== (0-4096), not real-world lat/lons.
	- The rendered knows the tile's geographic bounds and maps those integers back to screen pixels; thi is partially why the format is compact (no floats, no CRS metadata)

## Encoding
- The binary format is [[Protobuf|Protocol Buffer]]s; ==this is why [[Mapbox Vector Tile|MVT]] files are small compared to [[GeoJSON]]==


At rest, can be stored as:
- Individual files on disk/S3
- In a [[MBTile]]s file (SQLite database of tiles
- In a [[PMTiles]] file (single binary archive optimized for HTTP range requests, the vector equivalent of [[Cloud-Optimized GeoTIFF|COG]]s).
	- (Not sure if this holds, COGs are typically for one raster image of which you're loading a subset, while PMTiles has many tiles in it.)



