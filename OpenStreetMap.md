---
aliases:
  - OSM
---
The OSM project provides free [[Raster Tile]]s at `tile.openstreetmap.org`
- These are pre-rendered PNG images at each zoom level; the classic [[Slippy Map]] tiles that powered Google Maps-style web maps before [[Vector Tile]]s existed.

Fundamental limitation
- Raster tiles are images, not data. 
	- [[MapLibre GL JS]] can render them, but you lose everything that makes vector tiles powerful:
		- You can't change the style
		- Can't change the label language
		- Can't animate
		- Can't control layer ordering
	- The hex layer would sit on top of a fixed image, rather than integrating into the map.
- Also, OSM's usage policy proohibits heavy use in applications; they're met for low-traffic OSM ecosystem projects, not production applications.

Tradeoffs:
- +: Free, no key, no signup
- -: Raster, can't customize style to complement hex layer
- -: Usage polixy prohibits production use
- -: Lower resolution appearance vs vector tiles