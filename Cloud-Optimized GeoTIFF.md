---
aliases:
  - COG
---
See also: [[Tagged Image File Format|TIFF]], [[GeoTIFF]]

References:
- [{CNG 2025} Is COG Scalable - Jeff Albrecht](https://youtu.be/-9tNOBRidVA)

A [[GeoTIFF]] with a specific internal organization designed for efficient remote access.
- Often paired with [[SpatioTemporal Asset Catalog|STAC]]s

A [[Raster]] format designed for efficient HTTP range requests -- you can fetch just a tile-sized chunk of a massive satellite image without downloading the whole thing.

This is essential for serving satellite imagery.
- Tools like [[TiTiler]] (a [[Tile Server]]) consume COGs directly.

Two key internal features:
- ==Internal Tiling==: Instead of storing rows left-to-right, top-to-bottom (strip layout),  the image is divided into tiles (e.g. 256x256 or 512x512 pixels). You can read one tile without touching the rest of the file.
- ==Overview Levels== (Pyramids): Lower-resolution versions of the image are embedded in the same file; When you request a spatial window at a given resolution, the COG reader picks the overview level closest to your target resolution and reads from that.
	- Each zoom level is typically 2x downsampled from the previous (a factor of 4 in pixel count).
	- The overviews are stored inside the same file; because of COG's ==header-first layout==, the reader knows exactly where each overview level lives before reading any pixel data.
	- ==One HTTP request gets you to the index, and then targeted range requests get you exactly the tiles you need.==


Request: "Give me zoom level 8 view of this scene."
- Read header: Find overview 3 offset
- Read only the specific tiles within overview 3 that intersect the viewport



