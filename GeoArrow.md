The [[Apache Arrow]] project provides a standardized language-independent columnar memory format. It enables shared computational libraries, zero-copy shared memory and streaming messaging, interprocess communication, and is supported by many programming languages and data libraries.


Spatial information can be represented as a collection of discrete objects using points, lines and polygons (i.e., [[Vector]] data). 
- The ==Simple Feature Access== standard provides a widely used abstraction, defining a set of ==geometries==: Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection. 
- Next to a geometry, simple features can also have ==non-spatial attributes== that describe the feature.

The ==GeoArrow== specification defines how the [[Vector]] features (geometries) can be stored in [[Apache Arrow]] (and Arrow-compatible) data structures.


## Relationship to [[GeoParquet]]
- The GeoParquet specification originally started in this repo, but moved into its own, leaving this repo to focus on the Arrow-specific specifications (Arrow layout and extension type metadata). 
- Whereas GeoParquet is a file-level metadata specification, GeoArrow is a field-level metadata and memory layout specification that applies in-memory (an Arrow array), on disk (using Parquet readers/writers provided by an Arrow implementation), and over the wire (using the Arrow IPC format).