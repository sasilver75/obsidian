A binary [[Vector]] data format for geographic features, designed to be fast, simple, and cloud-friendly. It's ==essentially a binary encoding of a GeoJSON-like feature collection==, built on [[FlatBuffers]], one of Google's serialization libraries.
- Sweet spot for FlatGeobuf is large, read-heavy, spatially-queried vector datasets (building footprints, road networks, admin boundaries) stored in [[Blob Storage|Object Storage]], or anything where you want fast bbox queries without a [[Tile Server]].
- It's increasingly the preferred format for distributing large open vector datsets (e.g. [[Microsoft Building Footprints]]).

Comparisons:
- vs [[GeoJSON]]: FlatGeobuf is 5-10x smaller and faster to read, but not human readable.
- vs [[Shapefile]]: FlatGeobuf is a single file vs the notorious .shp/.dbf/.prj/.shx bundle, and has no 2GB file size limit, 10-character field name limit, or column type restrictions.
- vs [[GeoPackage]]: FlatGeobuf is append-only, single layer, no attribute indexing, while GeoPackage is SQLite and great for random access by attribute + transactions. FlatGeobuf wins on read spead and cloud access, while GeoPackage wins on flexibility.
- vs [[GeoParquet]]: FlatGeobuf is row-oriented and great for reading whole features, while GeoParquet is coluimnar and great for analytics.

Core design goals:
- Fast sequential reads: Scan through features as a stream without parsing overhead
- Cloud-native random access: Jump to a spatial subset without reading the whole file
- Simple sec: Easy to implement readers/writers in any language
- Lossless: Preserves all GEoJSON properties and geometry types exactly

How it's structured:
```
[magic bytes + version]
[header: CRS, geometry type, attribute schema]
[optional spatial index]
[feature data: packed binary features]
```
- The feature data is just tightly packed FlatBuffers records, with no framing overhead between features.
- Reading sequentially is extremely fast because  you're just scanning a flat binary array.
- The optional spatial index is an [[R-Tree]] packed into the file header area, ordered using a [[Hilbert Curve]]. Features are sorted by their Hilbert position so spatially-nearby featurs are physically nearby in the file.

Weaknesses:
- No attribute indexing: You can't efficiently filter by NON-SPATIAL attributes without scanning everything.
- Append-only: no in-place edits, you have to rewrite the whole file
- Less tooling than Shapefile or Geopackage.

Spatial Index + HTTP Range Requests
- This is the key cloud-native feature, using the same pattern as [[Cloud-Optimized GeoTIFF|COG]]:
	- Request 1: Read header and get spatial index
	- Request 2: Query index for bbox -> Get byte offsets of matching figures
	- Request 3: Range request for just those bytes -> get features
- You never download features *outside* your area of interest; for a 10GB File covering the whole world, a query for a small city might transfer only a few KB.












