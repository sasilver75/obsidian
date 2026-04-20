---
aliases:
  - Lossy Compression
  - Lossless Compression
  - External Compression
  - Internal Compresion
---
An algorithm that makes data smaller, at the cost of having to encode data into the compressed format before saving, and having to decode data out of the compressed format before usage.
- In most cases, the benefits of smaller file sizes when stored outweigh the time it takes to encode and decode the compressed format.
- Can either be external or internal, and either lossy or lossless


# Lossy vs Lossless Compression
- ==Lossy Compression==: A type of compression where the exact original values CANNOT  be recovered after decompression, meaning the compression process permanently loses information. Tend to give smaller file sizes than Lossless Compression. [[Joint Photographic Experts Group|JPEG]], LERC.
- ==Lossless Compression==: A type of compression where the exact original values **can** be recovered after decompression. This means that the compression process does not lose any information. Lossless compression codecs tend to give larger file sizes than lossy compression codecs. [[gzip]], deflate, LZW, ZSTD


# External vs Internal Compression
- ==External Compression==: Compression not part of a file format's own specification, which is added on after the main file has been saved. Tends to be used as part of [[ZIP]] archives or with standalone [[gzip]] compression. Typically makes a file no longer [[Cloud-Native Geospatial|Cloud-Optimized]].
- ==Internal Compression==: Compression that's part of the file format's own specification. Formats like [[Cloud-Optimized GeoTIFF|COG]], [[Cloud-Optimized Point Cloud|COPC]], and [[GeoParquet]] include internal compression; useful for [[Cloud-Native Geospatial|Cloud-Optimized]] data formats, because it allows internal chunks to still be fetched with [[HTTP Range Request]]s, but have smaller sizes.
	- Typically, for files that have already been internally compressed, adding another layer of external compression ([[gzip]]) will not make the file smaller, and will reduce performance by requiring an extra decompression step before the data can be used.