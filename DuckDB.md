DuckDB is an ==in-process analytical database== (an ==embedded== SQL database), running inside your Python process, rather than as a separate server.
- It's free and open source (MIT LIcense)
- No open source dependencies.
- Primarily written in C++

==DuckDB is not an in-memory DB==! Similar to SQLite, one database is represented as one file, so it's very easy to share, back up, etc.
- Has [[ACID]] properties [[Transaction]].
- Multiple tables in one database
- Table columns are stored individually on disk
	- Allows for fast search with per-column statistics
	- Columnar compression: saves 3-5x on storage

==In-Process== Deployment model:
- No separate server
- No config, no env, no docker
- Minimal transfer delay/overhead; If you want to compute some big result in Postgres and you want to pull it into your Notebook, PG has to serialize the result into a binary encoding, ship it over the wire, deserialize it back in your client library, and then you can work with it.
	- Because DuckDB is in the same process, you can skip much of that.

![[Pasted image 20260413173026.png]]


You can run DuckDB:
- In the backend
- In the client
- In the browser (using WASM)
- On your phone
- In your watch

### Graceful Degradation
- What to dow hen out of RAM?
	- Almost all DuckDB operators "spill to disk" when they run out of ram, instead using disk space.
	- Should "never crash, always make progress."

### "Friendly SQL"
- Superset of the PostgreSQL dialect
- FROM-first syntax (if you want), trailing commas
- GROUP BY ALL, column selection
- Nested types, lambda functions
- Python-inspired
![[Pasted image 20260413173709.png]]

DuckDB + Python


### Vectorized Execution Model
- For any sort of query processing engine (database, cloud warehouse, dataframe), they work in one of three ways:
	- ==Row-at-a-Time== execution model: Iterate over all of the rows one at a time.
		- Classic DB approach; [[SQLite]], [[PostgreSQL|Postgres]], etc. work this way. 
		- Low memory footprint, come from a time when RAM memory is scarce so you couldn't fit your working set in memory at the same time.
		- Downside: High CPU overhead; lots of cycles not operating on data, but instead shuffling data around and synchronizing parts of your pipeline.
	- ==Column-at-a-Time==: Basically how all dataframe libraries owrk; keep entire column in memory at a time, and can iterate through it or use [[SIMD]] instructions to implement operations even faster.
		- Downside: You have a very large memory footprint, having to keep the column in memory, and any intermediate results have to be materialized for the whole column.
	- ==Vector-at-a-time== (==DuckDB== is this!, becoming trendy):
		- Best of both worlds! You operate on a *vector of rows* (a chunk of rows) at a time.
		- The trick is to optimize this vector size so that all of your current working rows fit into your CPU cache, so you don't even have to go to main memory to access it.
		- DuckDB is multi-threaded, so it's natural to parallelize processing over vectors (chunks). Parallelism is increasingly important -- even consumer grade laptops have like 18 core CPUs.
		- DuckDB's execution here is optionally order-preserving, which gives similar dataframe semantics where you can slice/dice rows/tables and know that your input/output rows are going to be correlated.


## DuckDB Spatial: Geospatial extension
- DuckDB extensions are compiled code modules that are downloadable at runtime from DuckDB, and provide types, functions, operators, optimizastions, etc.
	- You can write your own if you're proficient in C++! But they provide all of hte ones that you should likely use in practice.
		- JSON, Postgres, AWS, Azure, MySQL, SQLite
- Philosophy: Anything that isn't essential to DuckDB should be an extension.

- DuckDB Spatial is an official extension that ofer the `GEOMETRY` type. (points, lines, polygons, multipolygons), modeled after [[PostGIS]].
- ==No runtime dependencies==, like DuckDB itself
	- Statically bundles [[GDAL]], GEOS, PROJ
	- Embed the entire default PROJ projection database, so DuckDB will recognize >3,000 of the most common [[Coordinate Reference System|CRS]] definitions out of the box.

- It's excellent for offline analysis, Jupyter notebooks, running SQL against [[Apache Parquet|Parquet]] files on disk, single-machine batch jobs, etc. It's fast at column-oriented aggregation.
- It's not a server though -- it has no concurrent multi-client access, no persistent network connection pool, and no long-running process that multiple clients can talk to simultaneously -- if you have two processes needing the same data, DuckDB's architecture fights you.





