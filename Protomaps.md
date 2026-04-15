A project/company by Brandon Liu, former [[Mapbox]] engineer, that rethinks how tiles are distributed.
- Traditional [[Tile]] serving requires a [[Tile Server]] that responds to thousands of HTTP requests (one per tile).
- Protomaps introduced [[PMTiles]], a ==single archive file containing all tiles for the entire planet==, designed so that ==HTTP range requests==  can fetch individual tiles directly from [[Amazon S3|S3]]/[[Cloudflare R2|R2]]/[[Content Delivery Network|CDN]] without any server-side tile serving software!

This is architecturally elegant!
- Upload one ~100GB file to [[Cloudflare R2]]
- MapLibre fetches tiles directly from it using range requests.
- No server to maintain, no per-request costs (R2 has no egress fees)
- Scales to any traffic level.

==The [[OpenStreetMap]] [[PMTiles]] file is published ~~monthly~~ DAILY by Protomaps and is free to download.==

Styling:
- PMTiles ==just give you the raw tile data, so you still need a style!==
	- Write your own or use an open source style designed for the [[OpenMapTiles]] schema.
		- Good dark options include `versatiles-colorful`, but none are as polished as commercial offerings out of the box.

Tradeoffs:
- +: Essentially zero ongoing cost; It costs ~$2/month to store tiles for the planet on R2, with no egress fees.
- +: Full control, no vendor dependency.
- +: Works offline, no API key management
- -: Some upfront setup; Download 100GB file, upload to R2, configure CORS
- -: Style quality requires more effort to get right
- -: Map data only updates when you redownload and reuplload the planet file (monthly cadence)
	- ==oops, it actually publishes them on a DAILY cadence, now!==


____________________
But Protomaps is broader than just PMTFiles -- it's a company built around the idea that map tile infrastructuer should be simple and cheap!
- PMTiles are the key invention, a ==single-file archive format for map tiles.==
	- Instead of a tile server that responds to 1000s of individual requests like `/tiles/14/3456/5678.pbf`, a PMTiles file stores all those tiles in one file with a clever internal index.
	- The client can fetch *just the bytes it needs* from this file using HTTP range requests; no server needed, just a static file host (S3, R2, even Github).

But Protomaps also:
- ==Publishes a daily global PMT file==, which covers the entire world, typically down to building-shape level of detail, and is designed to be used as a single static file for maps.
	- Size: The global file is quite large, typically around 100-130GB in size.
	- Usage: These daily files allow users to avoid the complexity of setting up serve-side map rendering, enabling "self-hosted" maps with minimal infrastructure.
- Provides ==map styles== designed for their tiles
- Has a ==hosted service== (protomaps.com) if you don't want to self host.
- Contributes tooling for building/working with PMTfiles.



