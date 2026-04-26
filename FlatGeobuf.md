Resources:
- [Cloud-Optimized Geospatial Formats Guide: FlatGeoBuf](https://guide.cloudnativegeo.org/flatgeobuf/intro.html)

A binary file format for geographic [[Vector]] data (points, lines, polygons). Unlike other cloud-optimized formats like [[Cloud-Optimized GeoTIFF|COG]] which build on the previous success of other datatypes, FlatGeobuf is a new format, designed from the ground up to be faster for geospatial data. Supports any vector geometry type defined in the [[Simple Features Access|OGC Simple Features]] specification, as well as more obscure types.

A binary [[Vector]] data format for geographic features, designed to be fast, simple, and cloud-friendly. It's ==essentially a binary encoding of a GeoJSON-like feature collection==, built on [[FlatBuffers]], one of Google's serialization libraries.
- Sweet spot for FlatGeobuf is large, read-heavy, spatially-queried vector datasets (building footprints, road networks, admin boundaries) stored in [[Blob Storage|Object Storage]], or anything where you want fast bbox queries without a [[Tile Server]].
- It's increasingly the preferred format for distributing large open vector datsets (e.g. [[Microsoft Building Footprints]]).

Comparisons:
- vs [[GeoJSON]]: FlatGeobuf is 5-10x smaller and faster to read, but not human readable.
- vs [[Shapefile]]: FlatGeobuf is a single file vs the notorious .shp/.dbf/.prj/.shx bundle, and has no 2GB file size limit, 10-character field name limit, or column type restrictions.
- vs [[GeoPackage]]: FlatGeobuf is append-only, single layer, no attribute indexing, while GeoPackage is SQLite and great for random access by attribute + transactions. FlatGeobuf wins on read spead and cloud access, while GeoPackage wins on flexibility.
- vs [[GeoParquet]]: ==Both are for vector data. FlatGeobuf is row-oriented and great for reading whole features, while GeoParquet is coluimnar and great for analytics.==

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

![[Pasted image 20260425202144.png|1840]]


![[Pasted image 20260419015921.png]]
Source: https://guide.cloudnativegeo.org/overview.html#/flatgeobuf 
- Has four sections:
	- Magic Bytes (signature): 8 magic bytes indicating file type and spec version, so you know it's a FlatGeobuf, even if it's missing a file extension.
	- Header: Stores the ==bounding box== of the dataset, the geometry type of the features, the attribute schema, the number of features, and [[Coordinate Reference System|CRS]] information.
	- Index: ==Optional==; if included, lets a reader skip reading features that aren't in a provided spatial query. Uses a [[R-Tree|Hilbert R-Tree]]
	- Data (features): A sequence of the individual feature records, placed end-to-end.
- Features are laid out in a row-oriented fashion, rather than a column-oriented fashion. This means it's relatively cheap to select specific records from the file, but relatively expensive to select a specific column.
- No internal compression. A compression like gzip can be applied to the file in full, but keep in mind that storing the compressed file will eliminate random access support.
- FlatGeobuf is a write-only format, and ==doesn't support appending==, as that would invalidate the spatial index in the file.
- ==Supports streaming==, meaning you can use part of the file before the entire file has finished downloading. Can be valuable because it makes an application seem more responsive (see [this example](https://observablehq.com/@bjornharrtell/streaming-flatgeobuf))







