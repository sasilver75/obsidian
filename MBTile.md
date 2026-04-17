A ==specification for storing map tiles in a single [[SQLite]] database file==.

[[Mapbox]] created it as a way to package up thousands (or millions) of independent tile files into one portable, manageable file.

Structure:
- A [[SQLite]] database with a standard schema
- `tiles` table: Stores each tile keyed by (zoom_level, tile_column, tile_row) (z,x,y)
- Tiles are stored as raw PNG, JPEG, or gzipped [[Mapbox Vector Tile|MVT]].


==Unlike [[PMTiles]]==, ==can't be served from [[Blob Storage|Object Storage]].==
- SQLite isn't a network protocol, it's a file format that requires random access reads at the byte level, managed by the SQLite engine, which has to run in a Python process.

Common MBTiles Workflow:
- Generate tiles with tools like [[Tippecanoe]] (Vector) or [[GDAL]] (raster).
- Host with a tiler server like [[Martin]], [[pg_tileserv]], etc.
- Tile server translates /{z}/{x}/{y} requests into SQLite queries

Strengths:
- Extremely mature ecosystem
- Easy to inspect/debug
- Great for offline use cases (mobile apps, field work)
- Widely supported by GIS tools ([[QGIS]], [[Mapbox]])


