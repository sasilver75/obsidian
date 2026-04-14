---
aliases:
---
Tile Pyramid:
- Web maps work by dividing the world into a grid of square image tiles at multiple zoom levels:
	- **Zoom 0**: the entire world in 1 tile (256×256 pixels)
	- **Zoom 1**: 4 tiles
	- **Zoom 2**: 16 tiles
	- ...
	- **Zoom 15**: ~1 billion tiles (street level detail)
	- **Zoom 20**: building-level detail
- Here, each tile is addressed by a ==zoom/x/y== -- this is the ==XYZ tile scheme==, also called a [[Slippy Map]] (a term referring to modern web maps which let you zoom and pan around). Jeff Albrecht refers to this as an "XYZ Tile".

![[Pasted image 20260411185613.png]]
Above: XYZ Tiles coordinate numbers. Z is the zoom level, and X and Y identify the tile. This is the de-facto [[OpenStreetMap]] naming standard, known as either "Slippy Map Tilenames" or "XYZ". 
- Typically served by API from URLS like `http://..../Z/X/Y.png`

Most tiled web maps follow certain Google Maps conventions:
- ==Tiles are 256x256 pixels==
- At the outer-most zoom level, 0, the entire world can be rendered in a single map tile.
- ==Each zoom level doubles in both dimensions, so a single tile is replaced by 4 tiles when zooming in==. This means that about 22 zoom levels are sufficient for most practical purposes.


## Raster Tile vs Vector Tile
- [[Raster Tile]]s: ==Pre-rendered== PNG/JPEG images. Simple to serve (static files via CDN), but not interactive. You can't change style, query feature properties, or filter on the client side. Used for satellite imagery *base maps*.
- [[Vector Tile]]s: Binary-encoded geometry + properties (`.pbf` / Mapbox Vector Tile format). The ==rendering happens in the browser.==
	- Styles can change without re-fetching tiles.
	- You can query features under a click.
	- You can filter features on the client side.
	- Tiles are typically much smaller than raster equivalents.

Things like MapLibre GL JS and Mapbox GL JS consume vector tiles; this is generally what you want for interactive layers.

[[pg_tileserv]] is a lightweight Go server that generates vector tiles directly from PostGIS queries. You point it at your database and it exposes your spatial tables as tile endpoints:

```
GET /public.service_requests/{z}/{x}/{y}.pbf
```

No pre-generation step: Tiles are computed on demand from live data. For a project where data updates daily, this is ideal! For very high traffic, you'd add a tile cache in front.

[[Martin]](written in Rust) is an alternative with more configuration options and perhaps better performance. Both of these are [[Tile Server]]s.

[[TiTiler]] is another common alternative.


