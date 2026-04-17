An open-source project that defines a ==schema and toolchain for converting [[OpenStreetMap]] data into [[Vector Tile]]s== that can be self-hosted or used as a [[Basemap]].

OSM planet.osm / regional extract
          ↓
  imposm3 (imports OSM PBF into PostgreSQL/PostGIS)
          ↓
  SQL layer definitions (one per theme: roads, buildings, water, etc.)
          ↓
  pg_tileserv or tegola (serves MVT tiles from PostGIS)
          ↓
  MapLibre GL + a style JSON (renders in browser)

The core contribution is the schema; a well-defined set of layers (transport, landcover, water, building, place labels, etc.) with consistent attributes.
- It's become the ==de-facto standard schema for OSM-based vector tiles==, and generates [[MBTile]]s.
	- ==[[MapTiler]], [[Stadia Maps|Stadia]], etc. all serve tiles compatible with the OpenMapTiles schema.==


Before OpenMapTiles, self-hosting a vector tile basemap required significant expertise.
- OMt packaged the whole pipeline into something reproducible.

License:
- OMT schema and styles are open, but the project moved to a dual license in 2019 (free for open-source, paid for commercial use); this is why [[Protomaps]] emerged as a fully open alternative: same idea (OSM -> PMTiles -> self-hostable), fully open license, and PMTiles format avoids needing any server process at all (just static file hosting on R2 or S3).
