
Resources:
- [[Brandon Liu]]: [Minimum Viable Cloud-Native Geo @ Cloud-Native Geospatial Forum](https://youtu.be/4Qn3fOMQ6fI)

The goal of PMTiles is a minimal-viable idea of reading datasets directly into the browser, which involves tradeoffs around latency and how much data you're transferring.

If you're going from an S3 bucket to a browser, you want to make certain tradeoffs to minimize how long it takes for web apps to load.

(Understanding the ==tradeoffs== that PMTiles make, and understanding when that's appropriate to use them, is very important.)

PMTiles is an opinionated spec for this access pattern:
- Assumes you want a [[Tile Pyramid]], which has multi-resolution overviews
- It assumes that you want interactivity, meaning you want to [[Tile]] the data, and clip to the viewport. 
	- This makes it inappropriate for al ot of cases related to analysis.
		- You can't give na accurate count of how any buildings are in a bounding box based on overviews, because tiles that are multi-resolution will drop some of those features as you move out.
		- That's a key difference when doing [[Vector Tile]]s, rather than working with [[Raster]] [[GeoTIFF]]s, for instance. 

Vector tiles are a lossy techinque for lightweight, interactive maps.
- Tiles are necessary for problems like
	- If you're zoomed in on just LA, and your polygon is the entire boundary of CA, you don't want to download 20MB of CA boundary to show that... So you need to do some viewport clipping.
- Tiles aren't a silver bullet but are very good for building interactive web miles.
- PMTiles are an convenient way to wrap up hundreds or thousands of tiles into a single cloud-native archive file.

***==Hierarchical==*** directory structure (usually two levels of directories, so you don't have to upfront download multiple MB of offsets)

Has a [[Run-length Encoding]] of repetitive data (e.g. ocean is 70% of the world, large area features).

- PMTiles uses a [[Hilbert Curve]]-based indexing scheme, like [[S2 Geometry|S2]]
	- Hilbert curves defer the issue of "optimal block size" to readres, instead of writers; every individual fractal edge in the Hilbert Curve is itself a block, so the reader can be left to decide this for themselves.
- Uses a single 2D face ([[Mercator|Web Mercator]]) like [[Bing QuadKey]]
- Tile indexes start at 0 and are contiguous; unlike S2 or QuadKey, the cell level cannot be derived by counting bits.

==Compressed Bitmaps== to store sets of tiles
- Because TileIDs are contiguous 64-bit integers, programs should represent them as bitmaps in memory.
- Roaring Bitmaps is the preferred compressed bitmap implementation of compressed bitmaps
- This means that an area of interest is a range of integers, or set of ranges
	- This means you can store very large tile sets (> 100M tiles) in less than 10MB.


Q: "What about Raster PMTiles?"
A: This is less interesting because COG is everywhere, but we do have a rio-pmtiles plugin that lets you create PMTiles from any [[GDAL]] source... but most audiences are going to want to use COG.


Ecosystem
- [pmtiles.io](pmtiles.io) Is a viewer for PMTiles resources that could be either remote or local.
- PMTiles CLI: Single-binary CLI to install; edit, extract, upload, serve PMTiles, works with Azure/S3/etc. buckets.
- pmtiles extract: You can extract a smaller tile set from a larger one
- overture-tiles: Create PMTiles from [[Overture Maps Foundation]] once a month.


Compare with:
- [[GeoParquet]]: A data format for vector features; columnar storage of points/lines/polygons with attributes, useful for building footprints, parcels, POIs.
- [[Cloud-Optimized GeoTIFF]] (COG): A data format for raster imagery that lets you read a spatial subset without downloading the whole file, useful for satellite imagery, elevation maps, etc.
... So they're complimentary technologies; a typical pipeline might store raw imagery as COGs, derived features as GEoParquet, and serve a stylized basemap via PMTiles.

