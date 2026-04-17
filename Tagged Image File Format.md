---
aliases:
  - TIFF
---
A high-quality [[Raster]] image file format from Adobe popular in photography, graphic design, and publishing, known for using lossless compression or no compression to preserve image data and detail, often resulting in large file sizes compared to other image file formats like [[Joint Photographic Experts Group|JPEG]].

The "Tagged" part is key: Rather than a fixed structure, a TIFF file is a collection of tags (metadata fields) that describe what the data is and how it's stored. This makes it flexible:
- Supports many data types: uint8, uint16, float32, etc.
- Supports multiple bands
- Supports many compression schemes: LZW, DEFLATE, PackBits, JPEG
- Can store multiple images (pages) in one file
- Widely supported everywhere
