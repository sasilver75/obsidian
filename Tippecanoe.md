An open-source GIS ==command-line tool== originally built by [[Mapbox]] (now maintained by Felt) ==designed to build [[Vector Tile]] sets== (like [[MBTile]]s, maybe [[PMTiles]]) ==from large [[GeoJSON]], Geobuf, or CSV datasets==.

Its main job is deciding what to show at each zoom level -- simplifying geometries, dropping features, clustering points -- so tiles don't get too large.
- [[Mapbox Tiling Service]] (MTS) is Mapbox's cloud-hosted pipeline that essentially does the same thing, but runs on Mapbox's infrastructure instead of your machine.

Tippecanoe is the spiritual predecessor and open-source equivalent of MTS; maintenance has since been handed off Tippecanoe maintenance to [[Felt]], a mapping startup, so it's no longer a Mapbox project.


> "==The goal of Tippecanoe is to enable making a scale-independent view of your data==, so that at any level from the entire world to a single building, you can see the density and texture of the data, rather than a simplification from dropping supposedly unimportant features or clustering or aggregating them."
