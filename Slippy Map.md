---
aliases:
  - Tile Map
  - Tile Web Map
---
A web map displayed by seamlessly joining together dozens of individually-requested data files, called [[Tile]]s.

Each time the user pans, most of the tiles are still relevant and can be kept displayed, while new tiles are fetched. Data associated with irrelevant tiles can be excised.
- Individual tiles can be pre-computed, a task that's easy to parallelize.

Google Maps was one of the first major mapping sites to use this technique. The first tiled web maps used [[Raster]] Tiles, before the emergence of [[Vector]] Tiles.

![[Pasted image 20260411185334.png]]
Above: From [[OpenStreetMap]]

For a Slippy Map application, your data layers often have a base map underneath: Things like streets, neighborhoods, labels, etc. There are many options!
- MapTiler Cloud: Free tier, commercial service
- Stadia Map: Free tier for open source
- OpenStreetMap: Raster tiles (free but raster-only, so no custom styling)
- Self-hosted: Download [[OpenStreetMap]] data, generate tiles with `tilemaker`, serve with `mbtileserver`; more work, but more control.



## Frontend Map Libraries

#### MapLibre GL JS
- An open-source fork of Mapbox GL JS, which added a restrictive license in 2020 -- this is the community continuation.
- Renders [[Vector Tile]]s using [[WebGL]].

#### deck.gl
- A [[WebGL]]-powered visualization library from Uber, designed for large-scale data overlays on top of a base map. Handles millions of points efficiently.
- Integrates with MapLibre as an overlay, so you can get the basic map from MapLibre and the data layers from deck.gl -- they work together well!

