Resources:
- [VIDEO: 3b1b Essence of Linear Algebra: Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=1&pp=iAQB)

Vectors refer to quantities that can't be expressed by a single number. In Physics, they refer to quantities that have magnitude and direction (eg forces, velocities), while in Computer Science they're often thought of as ordered lists of numbers (eg tuples). Both geometric vectors can be added and scaled, leading to the concept of vector spaces.

They give data analysts a way to conceptualize lists of numbers in a visual way, and gives people like physicists a language to describe space, and the manipulation of space, using numbers that can be crunched and run through a computer.

![[Pasted image 20240603162041.png]]
Above: The coordinates of a vector are (eg) a pair of numbers that basically give instructions for how to get from the tail of the vector at the origin to its tip, with the first number telling you how far to walk along the x-axis, and the second number how far to walk parallel to the y-axis. You can continue to add dimensions/axes as needed, increasing the length of your vector representation.

We can add same-dimensioned vectors, multiply by scalars, and apply linear  transformations to vectors.


____________________

# In Geographic Information Systems
- Vector Data represents discrete ***geometric shapes*** with attributes attached, with three primary types:
	- **Point**: A single location (e.g. crime incident, 311 request, tree)
	- **Line/LineString**: Sequences of points forming a path (e.g. street segment, bus route, river)
	- **Polygon**: A closed ring of points forming an area (e.g. a neighborhood boundary, building footprint, park, census tract)

Vector data is what you get from city open data portals, census [[Shapefile]]s, and APIs like Yelp.

It's for discrete features with precise geometry: points, lines, polygons, and answers: "Where is this thing?"

### Common Vector Formats:

- [[Shapefile]] (`.shp` and friends)
	- The legacy standard, made by ESRI in the early 1990s. Every GIS tool on earth reads it.
	- It's not a single file, it's at minimum three files that travel together:
		- `.shp`: The geometry (coordinates)
		- `.dbf`: The attributes (like a spreadsheet)
		- `.shx`: An index linking geometry to attributes
	- Often also with:
		- `.prj`: Projection definition
		- `.cpg`: Character encoding
	- Problems:
		- Column named capped at 10 characters
		- No support for nested data or lists
		- 2GB file size limit
		- No standard for storing time or booleans
		- Split-file format
	- Commonly seen in government data downloads, but typically you only ant to use it when you have no choice; instead, use GeoPckage or GeoJSON.
- [[GeoJSON]]
	- A standardized human-readable geometry structure that's native to the web. Every mapping library ingests it directly, requires no special tooling to read or write.
	- Problems:
		- Verbose (a city with 100k parcels can be >50MB; all that JSON punctuation adds up)
		- No spatial index inside the file; reading features in a bounding box requires scanning the whole thing.
		- Coordinates are always [[WGS84]]; no other projections supported
		- Not suitable for large datasets.
	- When to use it:
		- Small datasets
		- API responses
		- Config files defining boundaries
- [[GeoPackage]]
	- A single SQLite database file with standardized geospatial tables.
	- Developed by the [[Open Geospatial Consortium]] (OGC) as the modern alternative to the Shapefile.
	- Upside:
		- Single file
		- No column name length limits
		- Supports multiple layers (vector and raster) in one file
		- Has spatial indexes baked in
		- Handles any projection, not just WGS84
		- No size limits
		- Full attribute types (dates, booleans, etc.)
	- Downsides:
		- Not yet as universally supported as Shapefile in legacy tools, and still file-based, so "not suited for concurrent writes or cloud-native access."
	- When to use:
		- Local data storage and exchange.
		- Loading reference data into a PostGIS database.
		- Replacing Shapefiles in workflows.
		- Distributing datasets as single file downloads.
- [[FlatGeobuf]] (.fgb)
	- A binary, flat-buffer format designed for streaming and cloud access.
	- All the geometry packed efficiently into a single file with a built-in spatial index at the beginning.
	- Because the spatial index is at the start of the file, a web browser or API can send a range request to just fetch the index, then a second range request to only fetch the features in a bounding box, without downloading the whole file.
	- When to use:
		- Serving vector data directly from a CDN or object storage (S3) without a tile server.
- [[Well-Known Text|WKT]] and [[Well-Known Binary|WKB]]
	- They're not file formats, but they're geometry encodings used inside other systems. WKBs are more compact but less human readable.
	- Extended WKT is WKT with an explicit [[SRID]]

