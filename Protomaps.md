A project by Brandon Liu, former [[Mapbox]] engineer, that rethinks how tiles are distributed.
- Traditional [[Tile]] serving requires a [[Tile Server]] that responds to thousands of HTTP requests (one per tile).
- Protomaps introduced [[PMTiles]], a ==single archive file containing all tiles for the entire planet==, designed so that ==HTTP range requests==  can fetch individual tiles directly from [[Amazon S3|S3]]/[[Cloudflare R2|R2]]/[[Content Delivery Network|CDN]] without any server-side tile serving software.

This is architecturally elegant!
- Upload one ~100GB file to [[Cloudflare R2]]
- MapLibre fetches tiles directly from it using range requests.
- No server to maintain, no per-request costs (R2 has no egress fees)
- Scales to any traffic level.

==The [[OpenStreetMap]] [[PMTiles]] file is published monthly by Protomaps and is free to download.==

Styling:
- PMTiles ==just give you the raw tile data, so you still need a style!==
	- Write your own or use an open source style designed for the [[OpenMapTiles]] schema.