If you had 1.5 million 311 service requests in LA and tried to load all of it in GeoJSON in the browser at once, you'd be downloading hundreds of MB of data, and the browser would spend seconds parsing it, and rendering would be slow or crash entirely.

Tile servers solve this by slicing the map into small, pre-defined chunks called [[Tile]]s, and serving only the tiles that cover what the user is currently looking at.
- At low zoom, you get a handful of coarse tiles.
- At high zoom (street level), you get many fine-grained tiles, but only for the small area on screen.

Every tile on a web map is addressed by three numbers: `{z}/{x}/{y}`
- z: Zoom level; 0 is the world in one tile, with each increment quadrupling the number of tiles, so 1 = 4 tiles, 15 = ~1B tiles
- x: column, left to right
- y: row, top to bottom

A URL that serves a tile might look like:
```
https://tiles.example.com/layer/{z}/{x}/{y}.pbf
```
Tools like [[MapLibre GL JS]] request tiles automatically as the user pans and zooms (you never construct tile URLs manually). You just tell MapLibre where the tile server is, and it handles the rest.

For [[Vector Tile]]s, geometry and attributes are typically encoded in a compact binary format called Mapbox a Tiles, using the `.pbf` extension (Protocol Buffer Format). The tile contains raw geometry data - the browser renders it using WebGL.

Base Map Tiles vs Data Map Tiles
- Base Map Tiles: Streets, labels, terrain, and background that give the map context. We don't generate these, we pull them from a third-party CDN (MapTiler, Stadia Maps, etc.). Tools like `MapLibre` can fetch them automatically based on a JSON file that describes all the layers.
- Data Map Tiles: Our own data served as vector tiles from a server like `pg_tileserv`. These are overlaid on top of the base map.

Tile Server Examples:
- [[pg_tileserv]] (in Go): Generally considered easier to configure, with native support for PostGIS functions and automatic layer detection. Only serves from a single [[PostgreSQL|Postgres]] database.
- [[Martin]] (in Rust): Faster and more versatile, supporting multiple Postgres databases, MBTiles, and PMTiles, making it ideal for high-traffic or mixed-source scenarios. Often cited as 2-3x faster in benchmarks than `pg_tileserv`.
- [[TiTiler]]: A FastAPI-based tile server for [[Cloud-Optimized GeoTIFF]]s -- it doesn't store rasters itself, it reads COG files from object storage (e.g. S3) and generates raster tiles on demand, slicing out just the pixels needed for each tile request.

So when using MapLibre, you register a source and a layer:
```javascript
// Register pg_tileserv as a vector tile source
map.addSource('sr-311-points', {
  type: 'vector',
  tiles: ['http://localhost:7800/public.sr_311/{z}/{x}/{y}.pbf'],
  minzoom: 14,   // don't request tiles below zoom 14
  maxzoom: 20,
});

// Render the source as circle markers
map.addLayer({
  id: 'sr-311-circles',
  type: 'circle',
  source: 'sr-311-points',
  'source-layer': 'sr_311',   // matches the table name in Postgres
  minzoom: 14,
  paint: {
    'circle-radius': 4,
    'circle-color': '#e74c3c',
    'circle-opacity': 0.7,
  },
});
```
... and then MapLibre handles everything else, requesting the right `{z}/{x}/{y}` tiles as the user pans, caching them in memory, cancelling in-flight request for tiles that have scrolled off screen.



## Tile Formats
A tile server converts raw data into tiles; a tile format is what the tiles look like!
- [[Mapbox Vector Tiles]] (MVT; .mvt/.pbf)
	- ==The dominant web vector tile format==. 
	- The map is divided into a grid of tiles at each zoom level (standard {z}/{x}/{y} scheme).
	- Each tile contains the vector features (geometries, properties) clipped and simplified for that tile's bounding box and zoom level.
	- Encoded as [[Protobuf|Protocol Buffer]]s (binary).
- [[PMTiles]]
	- A ==single-file archive== containing an entire pyramid of MVT (or raster) tiles, with a spatial index for efficient range requests.
	- Developed by Protomaps as an ==alternative to serving tiles from a live database==.
	- Instead of running a tile server, you precompute all tiles, pack them into a PMTiles archive, upload to S3/R2/CDN, and serve tiles as HTTP range requests from static storage, with no backend required for the tiles.
		- While live `pg_tilserv` tiles always reflect the current database, PMTiles are static snapshots and would have to be regenerated intermittently.


## Tile Server Deployments

![[Pasted image 20260413164125.png]]
This is what it typically looks like when you deploy a tileserve like [[TiTiler]]
