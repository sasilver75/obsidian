## A5
A Discrete Global Grid System that uses a dodecahedron as its base solid, chosen because it has the lowest vertex curvature of any Platonic solid, minimizing distortion when mapping to a sphere. It uses equilateral pentagons to tile the surface and achieves truly equal-area cells at every resolution level across 31 resolution levels, addressing the up-to-2x cell-size variation weakness of H3.

## Albedo
A measure of how much sunlight is reflected by a surface, expressed as a fraction from 0 (total absorption) to 1 (total reflection). Earth's average albedo is about 0.3; high-albedo surfaces like ice and clouds cool the planet by reflecting sunlight, while low-albedo surfaces like oceans and asphalt absorb more heat.

## Alembic
A lightweight database migration tool for Python, used alongside the SQLAlchemy toolkit. It tracks applied migrations in an `alembic_version` table and lets you define upgrade/downgrade SQL steps as versioned revision files, traversable with commands like `alembic upgrade head`.

## AlphaEarth
A Google DeepMind foundation model project aimed at mapping the planet in unprecedented detail using satellite imagery and AI.

## Amazon S3
Amazon Simple Storage Service (S3) is AWS's scalable object storage service, widely used as the backbone for data lakes, ML datasets, and cloud-native geospatial workflows such as hosting Cloud-Optimized GeoTIFFs and Parquet files.

## Amazon SNS
Amazon Simple Notification Service (SNS) is a managed pub/sub messaging service from AWS, commonly used to push real-time notifications — for example, alerting subscribers whenever a new GOES-R satellite file is dropped on S3.

## Analysis Ready Data
Satellite imagery that has been preprocessed to a standard allowing direct scientific or analytical use without additional radiometric or geometric correction. Typically means the data has been orthorectified, atmospherically corrected to surface reflectance, cloud-masked, and placed in a consistent coordinate system.

## Apache Airflow
An open-source platform for developing, scheduling, and monitoring batch-oriented data workflows, originally developed at Airbnb. Workflows are defined entirely in Python as Directed Acyclic Graphs (DAGs) composed of tasks with dependencies, and a web UI provides visibility and control over runs.

## Apache Arrow
An in-memory columnar data format and inter-process communication standard that allows different tools (Pandas, DuckDB, custom C++) to share tabular data as a pointer with zero copying or serialization. Arrow is a memory format, not a file format — Parquet is its complementary on-disk counterpart.

## Apache Avro
A row-based binary serialization format created in 2009, designed for write-heavy streaming workloads. It is schema-driven (schemas stored as JSON), supports schema evolution for forward/backward compatibility, and is heavily used with Apache Kafka.

## Apache Flink
A distributed stream processing framework that lets you build stateful, fault-tolerant pipelines over infinite data streams. It models computation as a dataflow graph of sources, operators, and sinks, and provides exactly-once processing guarantees via periodic checkpointing based on the Chandy-Lamport algorithm.

## Apache Gravitino
An open-source unified metadata lake that manages metadata across heterogeneous data sources and compute engines from a single layer. It is the vendor-neutral open-source alternative to Databricks Unity Catalog, supporting Spark, Trino, Flink, and Hive across relational databases, data lakes, and Iceberg tables.

## Apache Hive
A data warehouse infrastructure built on top of Hadoop that provides SQL-like querying (HiveQL) over large distributed datasets stored in HDFS or object storage. It popularized the concept of schemas on read and influenced the design of later table formats like Apache Iceberg.

## Apache Iceberg
An open table format from Netflix (2017) for large analytic datasets stored on object storage like S3, designed to make a directory of Parquet files behave like a database table. It adds ACID transactions, schema evolution, partition evolution, time travel, and hidden partitioning through a metadata bookkeeping layer on top of the actual data files.

## Apache Parquet
A columnar binary file format designed for efficient analytical queries on large tabular datasets, used internally by data warehouses such as BigQuery, Snowflake, and DuckDB. Key features include built-in compression (Snappy, Zstandard), schema metadata, and predicate pushdown to skip row groups without decompression.

## Apache Polaris
An open-source catalog for Apache Iceberg tables, providing a REST catalog API that allows multiple query engines to discover and govern Iceberg tables stored on object storage.

## Apache Sedona
A cluster computing extension for Apache Spark that adds native geospatial data types, spatial SQL, and spatial joins, enabling large-scale geospatial analytics beyond what fits on a single machine. Its creators founded Wherobots as a managed PaaS around the system.

## Apache Spark
A distributed data processing engine built at Berkeley in 2009, now the industry standard for large-scale batch analytics. It distributes a dataset across a cluster and uses lazy evaluation — building a query plan and optimizing it before execution — with APIs in Python (PySpark), Scala, Java, and R.

## ArcGIS
A commercial GIS software suite from Esri, the dominant company in enterprise GIS. It is more polished and deeply integrated with government and municipal workflows than open-source alternatives, but costs thousands of dollars per license; its legacy `.shp` Shapefile format is ubiquitous in government data portals.

## Band Count
The number of separate data layers (bands) a raster file contains, where each band stores one value per pixel. An RGB image has 3 bands, RGBA has 4, and hyperspectral satellite data may have 200+ bands each capturing a different wavelength of light.

## Batch Processing
A data processing model where work is performed on a finite, collected dataset at a scheduled or triggered time, rather than on a continuous stream. Frameworks like Apache Spark abstract away hardware failures, load balancing, and concurrency to enable large-scale batch jobs.

## Bing QuadKey
Microsoft's tile addressing scheme for Bing Maps, based on a QuadTree — a string of digits (0–3) that traces a path down the tree to identify a map tile, where string length equals the zoom level. It is a tile addressing scheme on a flat Mercator projection, not a true Discrete Global Grid System; it inherits Mercator distortions and is designed for map display rather than spatial analysis.

## BSD License
A family of permissive open-source software licenses that allow use, modification, and redistribution with few restrictions, requiring only attribution. The 2-clause "Simplified BSD" and 3-clause BSD are the most common variants, used by projects like the original Mapbox GL JS before its relicensing.

## Canadian Space Agency
The Canadian government agency responsible for the country's space program, notably operating the RADARSAT series of SAR Earth observation satellites.

## Celery
A Python distributed task queue that moves work out of synchronous request cycles into background worker processes via a message broker (typically Redis or RabbitMQ). It supports task chaining, scheduled tasks via Celery Beat, and concurrent execution across multiple workers.

## Cesium
A WebGL-based library for 3D globe rendering in the browser, focused on visualizing flight paths, satellite imagery, and terrain in a true 3D environment rather than a 2D slippy map.

## Cloud-Optimized GeoTIFF
A GeoTIFF with a specific internal organization — tiled pixel storage and embedded overview pyramids at multiple zoom levels — designed for efficient HTTP range requests so that only the needed spatial subset and resolution level must be fetched. Tools like TiTiler can serve web map tiles from a COG hosted on S3 without downloading the full file.

## Cloudflare R2
Cloudflare's object storage service for storing large amounts of unstructured data, notable for having no egress bandwidth fees unlike typical cloud storage providers. It is commonly used for data lakes, cloud-native application storage, and hosting PMTiles archives.

## Connection Pool
A cache of open database connections that are reused across requests, avoiding the ~5–10ms overhead of opening a new TCP connection to PostgreSQL on every query. SQLAlchemy manages this pool automatically, maintaining a configurable number of idle connections ready to check out.

## Coordinate Reference System
A framework that defines how coordinates map to locations on Earth, specifying both a datum (Earth's shape model) and a projection. Common CRS examples include EPSG:4326 (WGS84, used for storing GPS coordinates) and EPSG:3857 (Web Mercator, used for rendering web map tiles).

## D3
A JavaScript data visualization library that provides low-level control over SVG and Canvas rendering, widely used for custom charts, graphs, and data-driven documents. It is not a map renderer, but is often used alongside mapping libraries for time-series charts and detail panels; `Plot` is its higher-level successor.

## Dask
A Python parallel computing library that mirrors the Pandas and NumPy APIs, distributing computation across multiple cores or a cluster with minimal code changes. It excels at "medium data" (10 GB–1 TB) on a single machine or modest cluster, with a lower learning curve than Apache Spark.

## Data Lakehouse
An architecture that combines the low-cost, flexible storage of a data lake with the data management and query performance features (ACID transactions, schema enforcement, indexing) traditionally found in data warehouses. Technologies like Apache Iceberg and Delta Lake enable the lakehouse pattern on top of object storage.

## Data Warehouse
A centralized repository optimized for analytical queries on structured, historical data, typically with a predefined schema and strong consistency guarantees. Examples include Snowflake, BigQuery, and Redshift, all of which use columnar storage internally.

## Databricks Unity Catalog
Databricks' centralized metadata and governance layer for managing tables, files, models, and features across a Databricks lakehouse. It provides unified access control, data lineage, and a single catalog for Spark, SQL, and ML workloads; Apache Gravitino is an open-source vendor-neutral alternative.

## DBAPI
Python's DB-API 2.0 (PEP 249) is a specification that defines a standard interface — `connect()`, `cursor()`, `execute()`, `fetchone()`, `fetchall()` — that all Python database drivers must implement. This contract allows higher-level tools like SQLAlchemy to talk to any database without driver-specific code.

## deck.gl
Uber's GPU-powered WebGL data visualization framework that renders large data layers (millions of points at 60 fps) on top of a base map provided by libraries like MapLibre GL JS. Everything in deck.gl is an immutable `Layer` description; built-in layers like `H3HexagonLayer` understand geospatial index formats directly.

## Delta Lake
An open-source storage layer developed by Databricks that brings ACID transactions, scalable metadata handling, and time travel to data lakes on object storage. It stores data in Parquet files and uses a transaction log to track changes, similar in spirit to Apache Iceberg.

## Differential Interferometric Synthetic Aperture Radar
A specialized InSAR technique that removes the topographic phase component from radar interferograms to isolate and precisely measure surface displacements at millimeter scale between two acquisition dates. While InSAR provides elevation, DInSAR is used for monitoring motion such as subsidence, earthquake deformation, or volcanic inflation.

## Digital Elevation Model
A generic term for any raster dataset where each pixel stores an elevation value for that location. DEMs are the parent category encompassing both Digital Surface Models (which include all above-ground features) and Digital Terrain Models (bare-earth only).

## Digital Surface Model
A raster of elevation values representing the top surface of everything — buildings, trees, power lines, and bare earth. Raw LiDAR or photogrammetry produces a DSM first; subtracting the DTM from the DSM gives the height of objects above the ground.

## Digital Terrain Model
A raster of bare-earth elevation with buildings and vegetation filtered out, produced by classifying a point cloud to separate ground returns from above-ground returns. The difference between a DSM and DTM at any pixel is the height of whatever structure sits on the ground there.

## Discrete Global Grid System
A system that partitions the entire Earth's surface into a hierarchy of discrete, non-overlapping cells, each with a unique identifier. Major options include S2 (square-ish cells on a cube projection), H3 (hexagonal cells), A5 (pentagonal cells), and Geohash (rectangular cells on a flat projection).

## Docker
A platform that packages code and its entire runtime environment into portable, isolated containers that run identically on any machine. Docker Compose extends this to multi-container applications, and features like named volumes and bind mounts handle data persistence and hot-reload development workflows.

## Document Object Model
The DOM is a tree-structured, in-memory representation of an HTML or XML document that browsers expose as a programmable API. SVG graphics are part of the DOM — every `<circle>` or `<rect>` is a real DOM node — but this makes SVG impractical for rendering tens of thousands of map features, which is why GPU-based WebGL tools are preferred for large geospatial datasets.

## DuckDB
An embedded, in-process analytical SQL database that runs inside your Python process with no separate server, config, or Docker required. It uses a vectorized (chunk-at-a-time) execution model with columnar storage, spills to disk gracefully when RAM is exhausted, and can query Parquet and GeoParquet files directly; its spatial extension bundles GDAL, GEOS, and PROJ statically.

## Dymaxion Projection
A map projection invented by Buckminster Fuller in 1943 that unfolds the Earth onto an icosahedron (a 20-sided solid of equilateral triangles) and then flattens it. It is the geometric foundation of H3 because the icosahedron minimizes per-face distortion, distributing projection error more evenly across the globe than cylindrical projections.

## Electromagnetic Spectrum
The full range of electromagnetic radiation ordered by wavelength, from long-wavelength radio waves through visible light to short-wavelength gamma rays. All objects on Earth reflect, absorb, or transmit energy in wavelength-specific ways — each material has a unique spectral fingerprint — which is the physical basis for remote sensing and multispectral satellite imagery.

## EPSG Code
A numeric identifier (between 1024 and 32767) assigned by the EPSG Geodetic Parameter Dataset to uniquely identify coordinate reference systems, datums, projections, and related geodetic entities. EPSG:4326 identifies WGS84 geographic coordinates; EPSG:3857 identifies Web Mercator used by web map tiles.

## European Space Agency
The intergovernmental space agency of Europe, responsible for developing and operating programs including the Copernicus Earth observation programme and its Sentinel satellite series.

## Extensible Markup Language
XML is a text-based markup language that stores and transports structured data using custom tags arranged in a hierarchical tree. It is the basis for formats like SVG, GML (Geographic Markup Language), and older OGC web service responses.

## Fiona
A Python library for reading and writing vector geospatial file formats (GeoPackage, Shapefile, GeoJSON, etc.) using a clean API built on top of GDAL's OGR layer. It is used under the hood by GeoPandas for file I/O.

## GDAL
The Geospatial Data Abstraction Library, a foundational C library (combined with OGR for vector formats) that provides a unified API for reading and writing hundreds of raster and vector geospatial formats. It underpins nearly all geospatial software — rasterio, Fiona, PostGIS, QGIS, and more — and its command-line tools (`gdalinfo`, `ogr2ogr`, `gdalwarp`) are the universal translators of geospatial data.

## geemap
A Python package that provides a Jupyter-friendly interface to Google Earth Engine, enabling GEE's planetary-scale analysis within Python notebooks rather than requiring JavaScript.

## GeoAlchemy2
A SQLAlchemy extension that adds PostGIS geometry column support, allowing you to declare `Geometry(Point, 4326)` columns in SQLAlchemy ORM models and use spatial types transparently. It stores geometries internally as WKB hex strings and integrates with Shapely for serialization and deserialization.

## GeoArrow
A specification for encoding geospatial vector geometries (points, lines, polygons) in Apache Arrow's columnar in-memory format, enabling vectorized SIMD operations over geometry arrays without row-by-row WKB deserialization. GeoArrow is to Arrow what GeoParquet is to Parquet — they are complementary: GeoParquet for disk storage, GeoArrow for in-memory computation.

## Geocode
Geocoding is the process of converting a human-readable address into geographic coordinates (lat/lon); reverse geocoding converts coordinates back into a human-readable address. It is used in data ingestion pipelines when records have addresses but missing coordinates, and in search UIs to fly the map to a typed location.

## Geocoding Service
A service that exposes geocoding (address → coordinates) and reverse geocoding (coordinates → address) as an API. Examples include Nominatim (open-source, powered by OpenStreetMap data), Google Maps Geocoding API, and Mapbox Geocoding.

## Geographic Information Systems
Software systems for capturing, storing, analyzing, and visualizing geographic and spatial data. GIS encompasses spatial operations like point-in-polygon queries, spatial joins, buffering, and intersection, implemented in tools like PostGIS, QGIS, ArcGIS, and Python libraries such as GeoPandas and Shapely.

## geopandas
A Python library that extends Pandas DataFrames with a geometry column, enabling spatial operations like spatial joins, buffering, and coordinate reprojection directly on tabular data. It wraps Shapely for geometry operations and Fiona/pyogrio for file I/O, and reads Shapefile, GeoJSON, GeoPackage, and PostGIS connections.

## GeoParquet
An open-standard extension to Apache Parquet that adds geospatial vector data types (points, lines, polygons) by storing geometry as WKB bytes in a column with CRS and bounding-box metadata. It gives Parquet's columnar analytics performance to spatial data and is widely used for large open datasets like Overture Maps and Microsoft Building Footprints.

## Georeference
The process of aligning a digital image or raster dataset to a known geographic coordinate system using ground control points, enabling it to be accurately positioned on the Earth's surface. The process typically uses curve-fitting techniques to derive a transformation formula from identified control points with known locations.

## GEOS
A C library that implements the JTS (Java Topology Suite) computational geometry specification, performing 2D spatial relationship and overlay operations on vector features. All of PostGIS's spatial relationship functions (`ST_Intersects`, `ST_Contains`, `ST_Distance`, etc.) call GEOS under the hood; Shapely is essentially a Python wrapper around GEOS.

## Geostationary Orbit
An orbit at exactly 35,786 km altitude where the satellite's orbital period matches Earth's 24-hour rotation, causing it to appear stationary over a fixed point on the equator. This enables persistent coverage of a large fixed region but results in lower spatial resolution, higher communication latency, and no coverage near the poles; used by weather satellites like GOES-R.

## Geosynchronous Orbit
An orbit at 35,786 km where the satellite's orbital period matches Earth's rotation rate, but the orbital plane may be inclined above or below the equator. Geostationary orbit is a special case of geosynchronous orbit where the inclination is zero (directly above the equator); true intentional non-geostationary geosynchronous orbits are rare.

## GiST
Generalized Search Tree — a flexible PostgreSQL indexing framework that allows extension authors to plug in custom index logic. PostGIS implements an R-Tree inside GiST for geometry columns, enabling fast bounding-box overlap and distance queries; a two-step process first filters by bounding box (approximate), then rechecks with exact geometry.

## Global Navigation Satellite System
The umbrella term for all satellite-based navigation systems that provide global positioning. Major constellations include GPS (USA), GLONASS (Russia), Galileo (Europe), and BeiDou (China).

## Global Positioning System
The United States' satellite-based navigation system, part of the GNSS family, consisting of a constellation of satellites in Medium Earth Orbit (~20,200 km) that provide global positioning, navigation, and timing data. GPS coordinates are natively expressed in the WGS84 datum.

## Globalnaya Navigazionnaya Sputnikovaya Sistema
GLONASS is Russia's satellite navigation system, the Russian counterpart to GPS, operating a constellation of satellites at ~19,100 km altitude in Medium Earth Orbit. Like GPS, it is one of the four fully operational Global Navigation Satellite Systems.

## GOES-R
The current generation of NOAA's geostationary weather satellites, including GOES-16 (GOES-East) and GOES-18 (GOES-West). Their Advanced Baseline Imager captures 16 spectral bands at 500m–2km resolution, with full-disk imagery every 10 minutes; the data is freely available on AWS S3 with SNS notifications.

## Google Cloud Storage
Google Cloud's object storage service (GCS) for storing large amounts of unstructured data, analogous to Amazon S3. It is used as the back-end for Google Earth Engine's data catalog and for cloud-native geospatial workflows on Google Cloud.

## Google Earth Engine
A cloud computing platform from Google for planetary-scale geospatial analysis, providing a vast catalog of satellite imagery (Landsat, Sentinel, MODIS, etc.) going back decades and a cloud computing environment — with JavaScript and Python APIs — to run analysis directly on that data without downloading it.

## Google Open Buildings
A Google dataset of building footprint polygons derived from satellite imagery using machine learning, providing global coverage particularly strong in Africa, Asia, and Latin America. It is distributed as GeoParquet files and used for humanitarian, urban planning, and population estimation applications.

## GPU Rendering Pipeline
The sequence of programmable stages the GPU executes to render geometry: vertex data is uploaded to the GPU, a vertex shader transforms coordinates (e.g., lat/lon to screen pixels), the GPU rasterizes triangles into pixel coverage, a fragment shader assigns colors (e.g., a count-based color ramp), and the result is written to the framebuffer. Libraries like MapLibre GL JS and deck.gl expose this pipeline through WebGL.

## Ground Sample Distance
The real-world size represented by one pixel in a satellite or aerial image, the most precise measure of what is loosely called "resolution." WorldView-3's 0.31 m GSD means each pixel covers a 31 cm × 31 cm square on the ground; GSD increases slightly when the satellite tilts off-nadir.

## H3
Uber's hierarchical hexagonal geospatial indexing system that divides Earth's surface into hexagonal cells at 16 resolution levels, each cell identified by a unique 64-bit integer (displayed as a hex string). Hexagons have exactly six equidistant neighbors, making spatial smoothing and neighbor queries more consistent than square grids; cells nest with a 1:7 parent-child ratio.

## Hilbert Curve
A space-filling curve constructed by recursively subdividing a square into four quadrants connected in a U-shape, rotated and reflected to maintain continuity, until the path visits every point in the plane. Its crucial property for GIS is locality preservation — points close together in 2D space tend to have nearby positions on the 1D curve — which is exploited by S2, H3, and PMTiles for spatial indexing.

## Ibis
A Python dataframe library that compiles Pandas-like expressions into SQL and executes them against a backend database or query engine (DuckDB, BigQuery, Snowflake, Postgres, etc.). The Ibis + DuckDB combination is popular for analyzing large geospatial datasets locally without spinning up a full PostGIS stack.

## InSAR
Interferometric Synthetic Aperture Radar is a remote sensing technique that combines two or more SAR radar images of the same area taken from slightly different positions or times, measuring the phase difference (interference pattern) between them to derive millimeter-scale surface topography or deformation. It is an all-weather, day/night mapping tool used for earthquake monitoring, volcanic inflation, and subsidence tracking.

## Interferometry
A measurement technique that exploits the interference of waves (electromagnetic, acoustic, or other) to extract precise information about distance, displacement, or surface properties. In the geospatial context, radar interferometry (InSAR) uses phase differences between SAR image pairs to measure ground deformation at millimeter precision.

## Iridium Satellite Constellation
A commercial satellite constellation of 66 active LEO satellites at 781 km altitude in near-polar orbits, providing global voice and data coverage to satellite phones and IoT devices, including polar regions. It is notable historically for the "Iridium flare" phenomenon, where the original satellites' reflective antennas produced brief, intensely bright flashes visible from the ground.

## Joint Photographic Experts Group
JPEG is a widely-used lossy image compression standard that reduces file sizes by 10:1 to 20:1 through discrete cosine transform-based compression, achieving small files with manageable quality loss. It is ideal for photographs and web images but unsuitable for data that must be exactly preserved (e.g., scientific raster data).

## Kafka
Apache Kafka is a distributed event streaming platform used as a high-throughput message queue or stream processing backbone. It organizes data into topics partitioned across brokers, with producers appending records to immutable partition logs and consumers tracking their progress via offsets; it supports exactly-once semantics and is used by the majority of Fortune 100 companies.

## kepler.gl
Uber's web-based geospatial data exploration and visualization application, built on deck.gl. It provides a rich UI for loading datasets, applying filters, and creating shareable map visualizations without writing code.

## Lance
An open columnar storage format designed for multimodal ML datasets, supporting images, video, audio, text, embeddings, and geometry in a single unified format. It offers 100x faster random access than Parquet for ML workloads, native hybrid search (vector + full-text + SQL), ACID transactions, zero-copy versioning, and integrations with Pandas, DuckDB, Polars, Spark, and Ray.

## Land Surface Temperature
A measure of how warm or cold Earth's surface is, derived from thermal infrared satellite data, capturing the energy exchange between land and the atmosphere. LST influences plant growth timing and is affected by surface albedo; it is a key variable for urban heat island studies and climate monitoring.

## Landsat
The longest-running Earth observation satellite program, a joint NASA/USGS effort dating to 1972, providing multispectral imagery of Earth's land surfaces with a 16-day revisit cycle. Landsat data, used for agriculture, geology, forestry, and land-use mapping, is freely available to the public.

## Leaflet
A lightweight, DOM-based (SVG) open-source JavaScript library for interactive web maps. It is well-suited for hundreds to a few thousand features, but its non-GPU rendering makes it impractical for the massive datasets handled by WebGL-based libraries like MapLibre GL JS or deck.gl.

## leafmap
A Python package for interactive geospatial mapping and analysis in Jupyter notebooks with minimal code, wrapping tools like GeoPandas, pydeck, and whitebox-tools. It includes 500+ tools and a no-code UI for loading and analyzing vector and raster data, but is intended for EDA and demos rather than production applications.

## Light Detection and Ranging
LiDAR is an active remote sensing technique that emits laser pulses and measures the time for reflections to return, producing precise 3D point clouds of the Earth's surface and above-ground objects. It is the primary method for generating high-accuracy Digital Surface Models and Digital Terrain Models.

## Lonboard
A Python library for visualizing large geospatial datasets interactively in notebooks, built on GeoArrow and GeoParquet for zero-copy data transfer and deck.gl for GPU-based rendering. The name is a play on "deck" (as in deck.gl) and "lon" (longitude), evoking a fast geospatial skateboard.

## Low Earth Orbit
The orbital regime from ~160 to 2,000 km altitude, where satellites complete an orbit in ~90–120 minutes and make up ~88% of all spacecraft. LEO provides high spatial resolution and low signal latency, but any single satellite revisits a given location only once or twice per day unless part of a large constellation.

## Mapbox
A mapping platform that created the modern web mapping stack — including the Mapbox GL JS library, the Mapbox Vector Tile specification, and the map style specification still used by MapLibre. In 2020 they relicensed Mapbox GL JS v2 from open-source to proprietary, triggering the community fork that became MapLibre; their tile service and cartography remain widely regarded as the highest quality available.

## Mapbox Vector Tile
The dominant binary vector tile format for web maps, encoded as Protocol Buffers and served via the standard {z}/{x}/{y} tile scheme. Each tile contains vector features (geometries and properties) clipped and simplified for its zoom level and bounding box; now an open OGC standard.

## MapLibre
An open-source organization formed in 2021 after Mapbox relicensed its GL JS library, maintaining the MapLibre GL JS browser library and the MapLibre Native mobile library. MapLibre continues to develop the open-source fork under the original BSD-style license.

## MapLibre GL JS
An open-source WebGL-based JavaScript library for rendering interactive vector tile maps in the browser with GPU-accelerated performance. It fetches vector tiles (typically MVT format), applies a JSON style specification to render base map layers, and handles camera controls (pan, zoom, tilt); it is used alongside deck.gl which renders data layers on top.

## MapTiler
A Swiss geospatial company and one of the primary corporate sponsors of MapLibre GL JS, offering tile hosting, the OpenMapTiles open schema, and map style tools. They create and maintain the OpenMapTiles schema used by most non-Mapbox tile providers including Stadia Maps and Protomaps.

## Materialized View
A database object that physically stores the result of a query as a snapshot, unlike a regular view which re-executes its query on every access. In PostgreSQL, materialized views require explicit `REFRESH MATERIALIZED VIEW` commands (with `CONCURRENTLY` to avoid locking reads) since there is no incremental refresh.

## Maxar
A commercial satellite imagery company operating a constellation of very-high-resolution (VHR) optical satellites, with WorldView-3 producing the sharpest commercially available imagery at 0.31 m GSD. Maxar is a major NGA contractor and operates an Open Data Program releasing free imagery after major natural disasters.

## MBTile
A specification for storing map tiles (PNG, JPEG, or gzipped MVT) in a single SQLite database file, originally created by Mapbox. Unlike PMTiles, MBTiles requires a SQLite-capable tile server (e.g., Martin, pg_tileserv) to translate HTTP tile requests into SQLite queries and cannot be served directly from object storage.

## Medium Earth Orbit
The orbital regime from ~2,000 to 35,786 km altitude, used almost exclusively for navigation/GNSS constellations including GPS (20,200 km), GLONASS (19,100 km), Galileo (23,222 km), and BeiDou. MEO provides large coverage footprints per satellite and predictable, consistent orbits.

## Mercator
A cylindrical map projection dating to 1569, where the Web Mercator variant (adopted by Google Maps in 2005) became the de facto standard for web mapping. It preserves local shape (conformal) but distorts area significantly at high latitudes — Greenland appears as large as Africa — and is used in EPSG:3857 for web map tile rendering.

## MetaTiling
A technique for logically changing the effective chunk size of a Cloud-Optimized GeoTIFF without physically rechunking the file. COGs store data in row-major tile order; MetaTiling lets a reader aggregate multiple physical tiles into a larger logical tile, which is important when the physical chunk size does not match the application's access pattern (e.g., temporal slicing of a time-series COG).

## Microsoft Building Footprints
A global dataset of building footprint polygons derived by Microsoft from satellite imagery using computer vision models, covering hundreds of millions of buildings worldwide. It is incorporated into the Overture Maps Foundation dataset and distributed openly for use in mapping, urban planning, and humanitarian applications.

## MIT License
A short, permissive open-source license that allows anyone to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software with minimal restrictions — only requiring attribution. DuckDB and many Python geospatial libraries use the MIT License.

## MODIS
The Moderate Resolution Imaging Spectroradiometer, a NASA satellite sensor aboard the Terra (1999) and Aqua (2002) satellites that measures Earth and climate variables across 36 spectral bands. Most bands have 1 km spatial resolution (with some at 250 m and 500 m), providing near-daily global coverage for land cover, fire, ocean color, and atmospheric monitoring.

## Mosaic
A composite image created by stitching together multiple overlapping images into a single, seamless, larger picture, commonly used in satellite imagery to cover areas larger than a single scene. Cloud-free mosaics are often assembled from multiple acquisitions across time to fill gaps caused by cloud cover.

## National Aeronautics and Space Administration
NASA is the US federal agency responsible for the civilian space program, aeronautics research, and Earth science, operating major satellite programs including Landsat (with USGS), MODIS, and GOES (with NOAA). NASA provides large amounts of free Earth observation data through portals like EarthData.

## National Geospatial-Intelligence Agency
The NGA is a US Department of Defense and Intelligence Community agency that produces geospatial intelligence from satellite imagery and other sources to support national security. It maintains the WGS84 standard and is a major customer of commercial very-high-resolution imagery providers like Maxar.

## National Oceanic and Atmospheric Administration
NOAA is the US federal agency responsible for weather forecasting, climate monitoring, and ocean and atmospheric science, operating satellites including the GOES-R geostationary weather series and JPSS polar-orbiting satellites. NOAA data is broadly freely available, including real-time GOES-R data on AWS S3.

## National Reconnaissance Office
The NRO is a US government agency within the Department of Defense and Intelligence Community that designs, builds, launches, and operates classified reconnaissance satellites for the US federal government. It is one of the primary sources of classified satellite intelligence imagery for national security purposes.

## Near Infrared
The portion of the electromagnetic spectrum just beyond visible red light, at wavelengths of approximately 750–900 nm, invisible to human eyes but detectable by silicon camera sensors. Plants reflect NIR very strongly due to leaf cell structure while water absorbs it almost completely, making NIR the basis for vegetation indices like NDVI.

## Normalized Burn Ratio
A remote sensing index calculated as (NIR − SWIR) / (NIR + SWIR), ranging from −1 to 1, used to identify burned areas and quantify fire severity. High values (~0.1 to 1) indicate healthy vegetation; low or negative values indicate recently burned areas, bare soil, or low vegetation.

## Normalized Difference Built-up Index
A geospatial index calculated as (SWIR − NIR) / (SWIR + NIR) that maps and monitors built-up areas such as buildings, roads, and impervious surfaces. Positive values indicate built-up areas; negative values indicate vegetation.

## Normalized Difference Vegetation Index
A widely-used remote sensing metric calculated as (NIR − Red) / (NIR + Red), ranging from −1 to 1, that quantifies green vegetation density and health. Values of 0.6–0.9 indicate healthy dense vegetation; values near 0 indicate bare soil or stressed vegetation; negative values indicate water, clouds, or snow.

## Normalized Difference Water Index
A remote sensing metric used to detect and monitor water content — either surface water bodies (McFeeters: (Green − NIR) / (Green + NIR)) or vegetation moisture (Gao: (NIR − SWIR) / (NIR + SWIR)), both ranging from −1 to 1. Values above ~0.5 typically represent open water surfaces.

## Open Geospatial Consortium
An international voluntary standards organization that develops and maintains open standards for geospatial content and location services, including GeoPackage, WMS, WFS, and the Mapbox Vector Tile specification (now an OGC standard). The OGC baseline comprises over 80 standards used globally by governments, industry, and research.

## OpenGL Shading Language
GLSL is the C-like programming language used to write shaders — vertex shaders and fragment shaders — that run on the GPU in the OpenGL (and WebGL) rendering pipeline. MapLibre GL JS and deck.gl compile GLSL shaders that transform geographic coordinates to screen pixels and compute per-pixel colors for map features.

## OpenLayers
An open-source JavaScript library for displaying interactive maps in web browsers, slightly more full-featured than Leaflet and popular in enterprise and government GIS applications. It is not WebGL-native and does not match the performance of GPU-accelerated libraries like MapLibre GL JS for large datasets.

## OpenStreetMap
A free, openly-licensed, volunteer-built map of the entire world — essentially the Wikipedia of maps — containing vector data (nodes, ways, relations with tags) covering roads, buildings, land use, and points of interest globally. Data quality varies widely: dense urban areas in wealthy countries often rival commercial alternatives, while rural areas in developing regions may be sparse or outdated.

## Optimized Row Columnar
A columnar binary serialization format designed for the Hadoop ecosystem (Apache Hive, Spark), featuring excellent compression and predicate pushdown to skip entire row groups that don't match a filter. ORC is optimized for analytical, read-heavy workloads and is one of the file formats supported by Apache Iceberg alongside Parquet and Avro.

## Orbit
The curved path a satellite follows around Earth due to gravitational force, classified by altitude into Low Earth Orbit (160–2,000 km), Medium Earth Orbit (2,000–35,786 km), and High Earth Orbit (above 35,786 km, including geostationary at exactly 35,786 km). Orbital characteristics — altitude, inclination, and period — determine coverage, resolution, revisit frequency, and communication latency.

## Orthoimage
An aerial photograph or satellite image that has been geometrically corrected (orthorectified) to remove distortions from camera tilt and terrain relief, producing a planimetrically accurate image from which true distances can be measured. Creating an orthoimage requires a Digital Elevation Model to correct for the varying distance between the sensor and the ground.

## Overture Maps
A large-scale open geospatial dataset produced by the Overture Maps Foundation (backed by Amazon, Microsoft, Meta, and TomTom), covering places, buildings, transportation, administrative boundaries, and base land features with a consistent schema. It is distributed as GeoParquet files on S3 and aims to be a more consistent, schema-normalized alternative to OpenStreetMap.

## Panchromatic
A single broad spectral band covering the full visible light range (roughly 450–750 nm), capturing a black-and-white image at typically higher spatial resolution than multispectral bands. Panchromatic imagery is commonly fused with lower-resolution multispectral data (pansharpening) to produce high-resolution color images.

## Pandas
The foundational Python data analysis library providing the DataFrame data structure for tabular data manipulation, aggregation, and I/O. It is the basis for GeoPandas (which adds a geometry column) and inspired the APIs of Dask, Polars, and Ibis.

## pg_tileserv
A lightweight open-source tile server that serves vector tiles directly from a PostGIS database, translating {z}/{x}/{y} HTTP tile requests into spatial SQL queries. It is part of the Crunchy Data ecosystem and enables serving PostGIS table data as vector tiles without a separate data pipeline.

## pgRouting
An open-source extension for PostgreSQL and PostGIS that adds geospatial routing and network analysis functions, enabling shortest-path calculations (Dijkstra, A*), driving distance analysis, and traveling salesperson problems on road network graphs.

## pgvector
A PostgreSQL extension that adds a `vector` column type and approximate nearest-neighbor index methods (IVFFlat, HNSW), enabling similarity search over dense vector embeddings stored alongside relational data. It is used for semantic search, recommendation systems, and retrieval-augmented generation (RAG) directly within Postgres.

## Photogrammetry
The science of extracting 3D measurements and spatial information from 2D photographs, including stereophotogrammetry (using overlapping images from different angles to reconstruct 3D geometry) and photomapping (assembling rectified aerial photographs into cartographic products). Raw LiDAR or photogrammetry outputs are typically Digital Surface Models that must be further processed to produce Digital Terrain Models.

## PlaceKey
An API service that assigns a unique, universal identifier to every place on Earth, enabling address matching and dataset joining across sources that use different naming conventions or schemas for the same physical locations.

## Planet Labs
An SF-based Earth observation company (now called Planet) that operates the world's largest satellite constellation by count, including the Dove/PlanetScope cubesat fleet (~200+ satellites) providing near-daily global coverage at 3–4 m resolution, and the SkySat fleet for sub-meter tasked imagery.

## PlanetScope
Planet Labs' primary Earth observation constellation of ~200+ small "Dove" cubesats in sun-synchronous orbit, collectively imaging the entire Earth's landmass every day at 3–4 m multispectral resolution. Its defining advantage is the near-daily revisit rate, making it ideal for time-series analysis such as crop monitoring, deforestation alerts, and change detection.

## PMTiles
A single-file archive format for map tiles (vector or raster) designed to be served directly from object storage (S3, Cloudflare R2) using HTTP range requests, eliminating the need for a tile server. Tiles are indexed with a Hilbert curve-based scheme and run-length encoded; the global OpenStreetMap PMTiles file is published daily by Protomaps.

## Polar Orbit
An orbit with an inclination of approximately 80–90 degrees to the equator, causing the satellite to pass over or near both poles on each revolution. Used for Earth-mapping, reconnaissance, and weather satellites, often combined with sun-synchronous timing; launching into polar orbit requires more energy than low-inclination launches because it cannot leverage Earth's rotational velocity.

## Polars
A high-performance DataFrame library for Python and Rust, built from scratch in Rust for parallelism and speed, using Apache Arrow as its memory model. It is often significantly faster than Pandas for in-memory analytical queries and is a popular alternative for large CSV/Parquet processing on a single machine.

## PostGIS
A PostgreSQL extension that adds a `geometry` column type and hundreds of spatial functions, making it the standard for serious geospatial database backends. It provides two spatial types — `geometry` (planar, fast) and `geography` (spherical, accurate meters) — and uses GiST-indexed R-Trees for efficient spatial queries.

## PostgreSQL
A powerful, open-source relational database known for its extensibility (PostGIS, pgvector, H3, full-text search), ACID compliance, and support for complex queries. It is the most popular open-source relational database for production workloads and the foundation for geospatial backends through the PostGIS extension.

## PROJ
A C library that converts coordinates between Coordinate Reference Systems using the mathematical projection formulas for hundreds of known CRS definitions. GPS data in WGS84, city survey data in a local State Plane projection, and satellite imagery in UTM can all be brought into a common system using PROJ; PostGIS's `ST_Transform()` calls PROJ under the hood.

## Protomaps
A project and company by former Mapbox engineer Brandon Liu that introduced the PMTiles format — a single archive file containing all tiles for an area, served via HTTP range requests from object storage with no tile server required. Protomaps also publishes a daily global OpenStreetMap PMTiles file and provides open map styles.

## psycopg
The standard synchronous PostgreSQL driver for Python (PEP 249/DB-API compliant), required by Alembic's migration runner and usable with Celery workers. It is complemented by asyncpg, an async-only driver with no DB-API overhead, commonly used in async web frameworks like FastAPI.

## pydeck
A Python library that exposes deck.gl's GPU-powered visualization layers in Jupyter notebooks, enabling interactive large-scale geospatial visualizations from Python without writing JavaScript.

## pyogrio
A fast, vectorized I/O library for reading and writing geospatial vector formats (Shapefile, GeoJSON, GeoPackage, etc.) that serves as a faster drop-in backend for GeoPandas, replacing the older Fiona-based I/O with significantly improved read performance.

## QGIS
A free, open-source desktop GIS application used as a graphical alternative to ArcGIS for loading, inspecting, visualizing, and analyzing geospatial datasets without writing code. It can connect directly to PostGIS databases, read Shapefiles, GeoJSON, and GeoPackage files, and run spatial analyses interactively.

## QuadTree
A hierarchical spatial data structure that recursively subdivides a 2D area into four equal quadrants whenever a region contains more than a threshold number of points. It offers adaptive resolution (dense areas are subdivided more finely), but has been largely superseded in production databases by R-Trees and Geohashes; it was historically influential on modern spatial indexing concepts.

## R-Tree
A spatial index structure invented in 1984 that organizes geometries by grouping nearby minimum bounding rectangles (MBRs) into a hierarchy of parent bounding boxes. PostGIS implements an R-Tree inside GiST for geometry columns, enabling efficient point-in-polygon, proximity, and bounding-box queries by pruning branches whose MBRs cannot contain matching geometries.

## RADARSAT
A Canadian Earth observation satellite program overseen by the Canadian Space Agency, consisting of RADARSAT-1 (1995–2013), RADARSAT-2 (2007–present), and the RADARSAT Constellation Mission (2019–present, three satellites). The constellation uses C-band SAR and achieves a 4-day revisit period, enabling all-weather, day/night imaging for ice monitoring, disaster response, and land use mapping.

## RapidEye
A retired Earth observation constellation of five satellites, originally operated by RapidEye AG and later acquired by Planet Labs, which decommissioned them in 2020. The satellites operated at 630 km in sun-synchronous orbit capturing imagery at 5 m resolution across five bands including a distinctive red-edge band useful for vegetation analysis.

## Raster
A data model that represents continuous spatial phenomena as a regular grid of cells (pixels), where each cell holds one or more numeric values. Raster data includes satellite imagery (pixel = spectral values), elevation models (pixel = meters above sea level), and temperature grids (pixel = degrees Celsius); contrast with vector data, which represents discrete geometric features.

## rasterio
A Python library for reading and writing raster geospatial data (GeoTIFF, COG, and ~160 other formats via GDAL) with a clean, Pythonic API. It reads pixel values as NumPy arrays, supports reprojection, spatial clipping, and raster metadata inspection, and is the primary tool for working with satellite imagery in Python.

## Ray
A distributed Python computing framework for parallelizing arbitrary Python functions and actors across multiple cores or machines, using a `@ray.remote` decorator pattern. Unlike Spark and Dask (which focus on data transformations), Ray excels at heterogeneous ML workloads — distributed training, hyperparameter search, and model serving — via its Ray Train, Ray Data, and Ray Serve libraries.

## Remote Sensing
The acquisition of information about objects or phenomena without physical contact, generally through satellite or airborne sensors that detect electromagnetic radiation. Active sensors (LiDAR, SAR/RADAR) emit signals and measure returns; passive sensors (optical cameras, multispectral imagers) detect reflected sunlight or thermal emission; data is classified into processing levels 0–4 from raw to fully derived products.

## Run-length Encoding
A simple lossless compression technique that replaces consecutive runs of the same value with a single value and a count (e.g., AAAAAABBB → 6A 3B), effective when data contains long homogeneous runs. It is used in raster data for land cover masks, binary cloud masks, and as a building block inside more sophisticated formats like JPEG and PNG; PMTiles uses it for repetitive tile data like ocean areas.

## S2 Geometry
A Google library and Discrete Global Grid System that maps the Earth's sphere onto the six faces of a cube, then hierarchically subdivides each face into a grid of cells — each with a unique 64-bit integer ID — across 30 levels of subdivision. The 64-bit IDs enable spatial indexing via ordinary B-Tree indexes, and nearby cells have nearby IDs due to a Hilbert curve-based ordering; used internally by Google Maps and BigQuery.

## Satellite
An artificial object placed in orbit around Earth to perform Earth observation, communications, navigation, or scientific missions. Key terms of art include acquisition (capturing an image), tasking (directing a satellite to a target), revisit time (how often a location is imaged), swath width, ground sample distance, and orthorectification.

## Scalable Vector Graphics
An XML-based format for 2D graphics using geometric primitives (circles, rectangles, paths) described mathematically rather than as pixels, making it resolution-independent and styleable with CSS. SVG elements are real DOM nodes, which makes SVG impractical for rendering large numbers of map features (performance degrades past ~10,000 nodes); WebGL-based tools are preferred for geospatial datasets.

## SedonaDB
An open-source, single-node geospatial analytical database written in Rust, designed to treat spatial data as a first-class citizen rather than an extension. It offers columnar in-memory storage with spatial indexing, CRS tracking, Arrow format with zero serialization overhead, and spatial-aware query optimization including spatial join acceleration.

## Sentinel
A family of Earth observation satellite missions operated by ESA as part of the Copernicus programme, with Sentinel-2 being the most widely used for optical land monitoring. Sentinel-2 captures 13 multispectral bands at 10–60 m resolution over a 290 km swath with a ~5-day global revisit (using both 2A and 2B satellites combined), and all data is freely and openly available.

## Shapefile
A legacy ESRI vector format for geospatial features that is actually a collection of 3–7 files that must travel together (`.shp` for geometry, `.dbf` for attributes, `.prj` for CRS, `.shx` for index). Despite limitations (10-character column name cap, 2 GB limit, no booleans), it remains ubiquitous in government GIS data portals.

## Shapely
A Python library for creating, manipulating, and analyzing 2D geometric objects (points, lines, polygons) by wrapping the GEOS C library. It performs planar operations in raw coordinate space without knowledge of CRS or Earth curvature — distances are in coordinate units, not meters — so projection is needed for real-world measurements.

## Short-Wave Infrared
The portion of the electromagnetic spectrum at ~1,000–2,500 nm, beyond near-infrared, that requires specialized sensors (InGaAs or HgCdTe detectors). SWIR penetrates thin haze and smoke, is sensitive to soil and vegetation water content, differentiates mineral and rock types, and is the basis for the Normalized Burn Ratio; Sentinel-2 and Landsat include SWIR bands, making them analytically more powerful than RGB-only cameras.

## Space-filling Curve
A continuous 1D curve that visits every point in a 2D (or higher-dimensional) space, creating a mapping from a line to a plane that preserves locality — nearby points in 2D tend to have nearby positions on the 1D index. Classic examples include the Hilbert Curve, Z-Order (Morton) Curve, and Peano Curve; they underpin scalable spatial indexing in S2, H3, and PMTiles.

## Spatial SQL
SQL extended with spatial data types and functions (typically conforming to the OGC Simple Features standard), enabling geometric operations like intersection, containment, buffering, and distance calculation directly in database queries. PostGIS is the canonical implementation; Databricks, DuckDB Spatial, and Apache Sedona also support spatial SQL.

## SPOT
Satellite Pour l'Observation de la Terre — a French commercial optical Earth observation satellite system operated by Airbus Defence and Space (formerly Spot Image). The series has run from SPOT 1 (1986) through SPOT 7 (retired 2023); SPOT 6 remains active, capturing imagery at 1.5 m panchromatic and 6 m multispectral resolution from a 832 km sun-synchronous polar orbit.

## SQLAlchemy
A Python SQL toolkit and ORM with two layers: a Core layer for building SQL expressions programmatically and managing connection pooling, and an ORM layer that maps Python classes to database tables. It abstracts dialect differences between databases, integrates with Alembic for migrations, and is extended by GeoAlchemy2 for PostGIS geometry column support.

## SRID
A Spatial Reference Identifier — the integer a spatial database uses internally to identify a Coordinate Reference System. In PostGIS, SRIDs are set to match EPSG codes (so SRID 4326 = EPSG:4326 = WGS84), though other spatial databases may use different numbering schemes.

## STAC
The SpatioTemporal Asset Catalog — a JSON metadata standard for discovering and querying raster geospatial assets (satellite scenes, climate datasets) by spatial extent, time range, and properties. STAC + COG is the standard cloud-native stack for satellite imagery; major public catalogs include Microsoft Planetary Computer and Element84 Earth Search.

## Stadia Maps
A tile hosting service targeting the open-source mapping community (MapLibre, Leaflet, OpenLayers users) with tiles based on OpenStreetMap data. Their `Alidade Smooth Dark` style, originally designed by Stamen Design, is widely considered the best freely-available dark map style; the free tier offers 300k requests/month with no credit card required.

## Stream Processing
A data processing model where records are handled individually and continuously as they arrive from a source, in contrast to batch processing which operates on bounded datasets at scheduled times. Common patterns include stateless transformations, time-windowed aggregations, and multi-stream joins; Apache Flink and Apache Kafka Streams are the primary frameworks.

## Sun-Synchronous Orbit
A Low Earth Orbit subtype (~400–1,000 km altitude) where the orbital plane precesses at the same rate Earth orbits the Sun (~1°/day), ensuring the satellite always crosses the equator at the same local solar time on every pass. Consistent lighting across all acquisitions is critical for change detection; almost all optical Earth observation satellites (Sentinel-2, Landsat, PlanetScope, SPOT) use SSO.

## Surface Reflectance
Satellite-measured radiance that has been atmospherically corrected to represent the actual reflectance of the ground surface, removing effects of atmospheric scattering and absorption. Surface Reflectance is the appropriate product for any multi-date analysis or spectral index calculation; Top-of-Atmosphere (TOA) radiance includes atmospheric effects and is the uncorrected raw alternative.

## Synthetic Aperture Radar
A radar remote sensing technique where a satellite emits microwave pulses and records the backscattered energy, then mathematically synthesizes measurements taken from many positions along its flight path as if from a single enormous antenna, producing high-resolution imagery. SAR sees through clouds and darkness (radar doesn't require sunlight), detects surface roughness and moisture, and enables millimeter-scale deformation measurement via InSAR.

## Tagged Image File Format
A flexible, high-quality raster image format from Adobe based on a collection of descriptive tags (metadata fields) rather than a fixed structure, supporting many data types, multiple bands, multiple compression schemes (LZW, DEFLATE, JPEG), and lossless storage. GeoTIFF extends TIFF by embedding geospatial metadata; it is the parent format for the geospatial raster ecosystem.

## Tippecanoe
An open-source command-line tool (originally by Mapbox, now maintained by Felt) for generating vector tile sets (MBTiles or PMTiles) from large GeoJSON, Geobuf, or CSV datasets. Its main job is determining what to show at each zoom level — simplifying geometries, dropping or clustering features — so individual tiles stay within size limits while preserving the density and texture of the source data.

## TiTiler
A Python tile server for raster data (GeoTIFF, Cloud-Optimized GeoTIFF) that dynamically generates raster tiles on demand from COGs hosted on object storage. Pronounced "Tea Tiler," it is a common component in cloud-native satellite imagery serving pipelines alongside STAC catalogs and COGs.

## Top-of-Atmosphere
Raw satellite-measured radiance with minimal processing — it includes the effects of atmospheric scattering and absorption between the ground and the sensor. TOA is the starting point for atmospheric correction; Surface Reflectance is derived from it and is the appropriate product for scientific analysis and spectral index calculation.

## Trino
A distributed, fast SQL query engine (formerly PrestoSQL) designed for interactive analytics on large datasets across heterogeneous sources — data lakes on S3, SQL databases, and NoSQL systems — using standard SQL without requiring data movement or centralization. It is commonly used as the query layer in open lakehouse architectures alongside Apache Iceberg.

## United States Geological Survey
The USGS is the US federal science agency responsible for natural resources, natural hazards, and Earth science, co-operating the Landsat program with NASA and distributing free satellite, topographic, and geologic data through platforms like EarthExplorer.

## Vector
In GIS, vector data represents discrete geometric features as points, lines, and polygons with associated attributes. Common vector formats include Shapefile, GeoJSON, GeoPackage, and GeoParquet; in contrast to raster data (continuous pixel grids), vector data is precise at boundaries and answers "where is this thing?"

## Well-Known Binary
A compact binary serialization format for geometry objects (points, lines, polygons) that is more efficient to parse and store than the text-based WKT. PostGIS stores geometries internally as Extended WKB (EWKB), which also encodes the SRID; the hex strings visible when querying PostGIS directly are EWKB representations.

## Well-Known Text
A human-readable text serialization format for geometry objects (e.g., `POINT(-118.24 34.05)`, `POLYGON(...)`) standardized by the OGC. WKT is used for inspecting PostGIS data, constructing geometries in SQL via functions like `ST_GeomFromText`, and in GIS desktop tools; WKB is its binary counterpart used for storage and transmission.

## WGS84
The World Geodetic System 1984 — the standard geodetic datum used for global positioning, navigation, and mapping, and the reference frame for GPS. Virtually every modern API and web mapping library returns coordinates in WGS84 (EPSG:4326), defined as an Earth-centered, fixed coordinate system with an associated ellipsoid model.

## whitebox-tools
An open-source geospatial analysis library and command-line toolset for advanced raster and terrain analysis, providing hundreds of tools for hydrological modeling, LiDAR processing, image analysis, and GIS operations. It is accessible from Python and used in leafmap for notebook-based geospatial workflows.

## Zarr
A chunked, compressed N-dimensional array format for cloud-native storage, where an array is split into many small independently compressed chunk files stored in a directory or object store prefix. Unlike COG (which handles 2D rasters), Zarr handles arbitrary dimensions — such as (latitude, longitude, time, variable) climate data cubes — enabling efficient partial reads via HTTP range requests on exactly the needed chunks; almost always used with the Xarray Python library.
