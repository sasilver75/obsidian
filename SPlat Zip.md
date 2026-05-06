---
aliases:
  - SPZ 4
  - SPZ
  - SPlat Zipped
---
References:
- Tweet: [Niantic Spatial's X post about SPZ 4](https://x.com/NianticSpatial/status/2051730615548977427?s=20), [Their Linked Blog Post](https://www.nianticspatial.com/blog/spz4?utm_content=377131687&utm_medium=social&utm_source=twitter&hss_channel=tw-631577690)

A file format for compressed [[3D Gaussian Splatting|3D Gaussian Splat]]s, which are typically around 10x smaller than the corresponding [[Polygon File Format|PLY]] files, with minimal visual differences between the two.

> "The [[Joint Photographic Experts Group|JPEG]] of Gaussian Splatting"

> "SPZ made it practical to share Gaussian splats directly from mobile devices, establishing a compact, portable format that enabled broad adoption."

> "SPZ is increasingly the connective tissue for Gaussian splats across major creative and developer tools"

> "SPZ 4 compresses about 3-5x faster, loads roughly 1.5-2x faster end-to-end, still produces files that are 10x smaller than uncompressed PLYs, and gives developers new controls over the quality so the same format can serve both maximum-fidelity and maximum-efficiency workflows."

> "SPZ 4 replaces [[gzip]] with six parallel [[Zstandard|zstd]] streams, one per attribute (positions, colors, scales, rotations, alphas, spherical harmonics). Each attribute is its own independently-compressed buffer, with a small table of contents up front so the parser/loader can quickly determine the size and layout of each compressed attribute stream (positions, colors, etc.) without having to decompress or scan the file first."

> "SPZ 4 puts the 32-byte file header in plaintext at a fixed offset, outside the compressed region. That means you can inspect a file's point count, SH degree, version, and flags without decompressing a single byte of payload data."

> "The other big architectural change in SPZ 4 is extensions... SPZ 4 introduces controlled extensibility, allowing new attributes and metadata to evolve within the format without breaking interoperability... \[An extension from Adobe is:] 0xADBE0002 Safe Orbit Camera stores recommended elevation and radius bounds for orbit-style camera controls, so splat viewers don't have to guess how to frame and navigate the scene."

