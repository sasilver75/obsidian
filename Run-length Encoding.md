---
aliases:
  - RLE
---

Run-length encoding (RLE) is one of the simplest ==compression== techniques.

Instead of storing repeated values individually, you store the value once plus a count of how many times it repeats.

==Example:==
Uncompressed: AAAAAABBBCCDDDDDD
RLE Encoded: 6A 3B 2C 6D

Instead of 17 characters, you store 8!

It ==works well when data has long runs of the same value==, as is the case for black and white images, simple graphics, binary masks (like compressing bitmaps).

In GIS, commonly in [[Raster]] contacts:
- Land cover/classification rasters: Large areas of the same class (ocean = 0, forest = 1, etc.) compress very well
- Binary masks: Cloud masks, water masks involve long runs of 0s and 1s
- [[Cloud-Optimized GeoTIFF|COG]]/[[GeoTIFF]] internals.: TIFF supports RLE (PackBits) as one of its compression options.
- Sparse data: Where there are lots of "nodata" values to compress to almost nothing.

It's also a building block inside more sophisticated compression schemes like [[Joint Photographic Experts Group|JPEG]] and [[Portable Network Graphics]] (PNG).














