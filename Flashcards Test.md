## 3D Tiles

> [!question]- Q: What is 3D Tiles and who developed it?
> A: 3D Tiles is an open OGC standard (since 2019) for streaming massive, heterogeneous 3D geospatial datasets — including photogrammetry meshes, LiDAR point clouds, BIM models, and terrain. It was developed by Cesium.

> [!question]- Q: What problem does 3D Tiles solve?
> A: It solves the problem of rendering enormous 3D datasets (e.g., a 500GB city mesh or trillions of LiDAR points) in real time, by enabling progressive level-of-detail loading: showing the whole dataset at low detail, streaming higher detail as the camera zooms in, and never loading data that isn't visible.

> [!question]- Q: What is a 3D Tiles "tileset" and how is it described?
> A: A tileset is the name for a 3D Tiles dataset. It is described by a `tileset.json` file that defines a hierarchical tree of tiles, each with a bounding volume, geometric error, content URI, and child tiles.

> [!question]- Q: What are the two refinement strategies in 3D Tiles?
> A: REPLACE (children completely replace the parent tile, used for photogrammetry meshes) and ADD (children add detail on top of the parent, used for point clouds).

> [!question]- Q: How does the Level of Detail selection algorithm work in 3D Tiles?
> A: It uses Screen Space Error (SSE), which projects a tile's geometric error into screen pixels. If SSE exceeds a threshold (~16 pixels), the tile is too coarse and its children are loaded; if SSE is at or below the threshold, the tile is rendered and subdivision stops. The entire visible tileset is traversed every frame.

> [!question]- Q: What tools beyond CesiumJS support 3D Tiles?
> A: deck.gl (Tile3DLayer), QGIS (experimental), Unreal Engine, Unity, and ArcGIS (Scene Viewer and ArcGIS Pro).

> [!question]- Q: What is the content format for tiles in 3D Tiles 1.1?
> A: In 3D Tiles 1.1, tile content is stored as glTF (GL Transmission Format) files.


## 3DEP

> [!question]- Q: What does 3DEP stand for and who runs it?
> A: 3DEP stands for 3D Elevation Program. It is run by the USGS (United States Geological Survey).

> [!question]- Q: What is the goal of 3DEP?
> A: To collect high-quality LiDAR coverage of the entire contiguous United States, with data freely available via the USGS National Map and OpenTopography.

> [!question]- Q: What data formats and resolutions does 3DEP provide?
> A: Point clouds are available as LAZ and EPT (Entwine Point Tiles) files. Derived DEMs (Digital Elevation Models) are available at 1m and 1/3 arc-second resolutions.

> [!question]- Q: Why is 3DEP considered significant?
> A: It is a genuinely transformative public dataset that would cost hundreds of millions of dollars to acquire commercially, yet it is freely available to the public.


## Acquisition

> [!question]- Q: What is an acquisition in the context of Earth Observation?
> A: An acquisition is the single instance of a sensor collecting data over an Area of Interest — one imaging event. When querying a data catalog (e.g., STAC, Google Earth Engine), you are searching for acquisitions matching specific criteria.

> [!question]- Q: What are the five steps encompassed when a satellite "acquires" an area?
> A: (1) The sensor is tasked to collect at a specific time/location, (2) the satellite may slew (change attitude) to point at the target, (3) data is collected during the overpass window, (4) raw data is downlinked to a ground station, and (5) data is processed into an image product.

> [!question]- Q: What key attributes define an acquisition?
> A: Datetime (when collected), geometry (view angle, sun angle, off-nadir angle), cloud cover percentage, resolution/imaging mode, and for SAR: polarization and incidence angle.

> [!question]- Q: What is the difference between systematic and tasked acquisition?
> A: Systematic satellites (e.g., Sentinel-1 and Sentinel-2) collect everything in their swath continuously on a fixed schedule. Tasked satellites (e.g., Maxar WorldView) are directed to a specific target by a customer or operator, making imaging time a finite, competed resource.


## Airbyte

> [!question]- Q: What is Airbyte?
> A: Airbyte is an open-source data integration platform that moves data from sources (databases, APIs, SaaS tools) to destinations (data warehouses, data lakes) through pre-built and custom connectors. It is the open-source alternative to commercial ELT tools like Fivetran and Stitch.

> [!question]- Q: Where does Airbyte sit in the ELT pipeline?
> A: Airbyte handles the Extract and Load steps — it moves data faithfully from source to destination without transforming it. Transformation is left to downstream tools like dbt.

> [!question]- Q: What are the four sync modes Airbyte supports?
> A: Full Refresh | Overwrite, Full Refresh | Append, Incremental | Append, and Incremental | Deduped. The last (Incremental | Deduped) is the most common production mode.

> [!question]- Q: How are Airbyte connectors implemented?
> A: Each connector is a Docker container that implements the Airbyte Protocol — a standard interface defining how to discover a schema, read records, and handle state for incremental syncs. Airbyte has 300+ pre-built connectors.

> [!question]- Q: What are the three deployment options for Airbyte?
> A: Airbyte OSS (self-hosted on Docker Compose or Kubernetes), Airbyte Cloud (managed SaaS), and PyAirbyte (a Python library for running connectors directly in Python without a server).

> [!question]- Q: How does Airbyte differ from dbt?
> A: They are complementary: Airbyte moves raw data into the warehouse (Extract + Load), while dbt transforms it once it's there. They are commonly used together.


## Albedo

> [!question]- Q: What is albedo?
> A: Albedo is a measure of how much sunlight is reflected by a surface, expressed as a fraction from 0 (total absorption) to 1 (total reflection).

> [!question]- Q: What are some key albedo values for common surfaces?
> A: Fresh snow is up to 0.85, Earth's average is 0.3, the Moon is 0.12, and water is less than 0.1.

> [!question]- Q: How does albedo relate to climate change?
> A: Albedo creates a feedback loop: melting ice reduces reflection, causing more solar absorption, which causes further warming. High-albedo surfaces (ice, clouds) cool the planet, while low-albedo surfaces (oceans, forests) warm it.

> [!question]- Q: Why do cities have higher temperatures related to albedo?
> A: Cities have low albedo due to asphalt and buildings, which absorb more heat rather than reflecting it — contributing to the urban heat island effect.


## AlphaEarth Foundations

> [!question]- Q: What is AlphaEarth Foundations and who made it?
> A: AlphaEarth Foundations (AEF) is a geospatial foundation model from Google DeepMind, released in September 2025. It produces dense 64-byte embedding vectors at 10m resolution covering Earth's land surface, released annually from 2017–2024 on Google Earth Engine.

> [!question]- Q: What data sources does AlphaEarth Foundations fuse?
> A: It fuses optical (Sentinel-2, Landsat), radar (Sentinel-1, PALSAR-2), LiDAR (GEDI), climate (ERA5, GRACE), elevation (Copernicus DEM GLO-30), and annotations (NLCD, Wikipedia, GBIF species records).

> [!question]- Q: What is the key innovation in AlphaEarth Foundations' temporal modeling?
> A: It is the first EO embedding approach to treat time as a continuous variable rather than discrete snapshots, enabling querying of embeddings for any date range, including interpolation and extrapolation beyond observed data.

> [!question]- Q: How does AlphaEarth Foundations perform relative to competitors?
> A: It achieves ~23.9% error reduction over the next-best method across 15 evaluations, outperforming Clay, Prithvi, SatCLIP, and CCDC/MOSAIKS — the first approach where no single competitor wins on any task.

> [!question]- Q: What is the model size and embedding compactness of AlphaEarth Foundations?
> A: The model is approximately 480M parameters. Each embedding is only 64 bytes — 16x more compact than the next most compact learned method.

> [!question]- Q: Is AlphaEarth Foundations open-weights?
> A: No. It is not open-weights; you cannot fine-tune the model itself, unlike models such as Prithvi 2.0. The embeddings dataset is available on Google Earth Engine.


## Amazon Ground Station

> [!question]- Q: What is Amazon Ground Station?
> A: Amazon Ground Station (also called AWS Ground Station) is a fully managed cloud service from Amazon Web Services that allows customers to contact, control, and downlink data from satellites using a global network of ground station antennas, without needing to build or manage their own ground infrastructure.

> [!question]- Q: What problem does Amazon Ground Station solve?
> A: Building and operating ground station infrastructure is expensive and geographically constrained. AWS Ground Station provides on-demand access to a global antenna network, so satellite operators can downlink data directly into the AWS cloud without owning any ground hardware.

> [!question]- Q: How does Amazon Ground Station integrate with AWS services?
> A: Downlinked satellite data is delivered directly into AWS services like Amazon S3, enabling immediate processing with EC2, Lambda, or other AWS tools without moving data off the cloud.


## American Society for Photogrammetry and Remote Sensing

> [!question]- Q: What does ASPRS stand for?
> A: ASPRS stands for the American Society for Photogrammetry and Remote Sensing.

> [!question]- Q: What is ASPRS?
> A: ASPRS is a professional organization in the United States focused on advancing the science and practice of photogrammetry, remote sensing, and geographic information systems. It publishes standards, technical guidelines (including LiDAR accuracy standards), and the journal Photogrammetric Engineering & Remote Sensing.

> [!question]- Q: What is one well-known standard published by ASPRS?
> A: ASPRS publishes the ASPRS Positional Accuracy Standards for Digital Geospatial Data, which define accuracy classes for elevation data (including LiDAR-derived DEMs) and are widely used in government and commercial mapping projects.


## Analysis Ready Data

> [!question]- Q: What does ARD stand for?
> A: ARD stands for Analysis Ready Data.

> [!question]- Q: What is Analysis Ready Data?
> A: Analysis Ready Data is satellite imagery that has been pre-processed to a level where it can be used directly for analysis without additional corrections. This typically includes radiometric calibration, atmospheric correction (converting to surface reflectance), geometric correction, and cloud masking.

> [!question]- Q: What problem does ARD solve?
> A: Raw satellite imagery requires significant preprocessing before it is scientifically comparable across time and space. ARD eliminates this burden for end users, enabling time-series analysis and change detection across different dates and sensors without custom preprocessing pipelines.

> [!question]- Q: Give an example of an ARD product.
> A: Landsat Collection 2 Surface Reflectance products from USGS are a canonical example of ARD — atmospherically corrected, consistently processed, and ready for direct scientific analysis.


## Antimeridian

> [!question]- Q: What is the antimeridian?
> A: The antimeridian is the 180-degree line of longitude — the line on the opposite side of Earth from the Prime Meridian (0°). It runs through the Pacific Ocean and roughly corresponds to the International Date Line.

> [!question]- Q: Why does the antimeridian cause problems in geospatial software?
> A: Geographic coordinate systems represent longitude from -180 to +180, making the antimeridian the seam where those two extremes meet. A geometry crossing this line (e.g., a bounding box around Fiji) gets nonsensical coordinates because the numbers wrap around, breaking spatial indexes, bounding boxes, and GeoJSON geometries.

> [!question]- Q: What is the standard solution for handling antimeridian crossings?
> A: The recommended solution (per RFC 7946 for GeoJSON) is to split a geometry at the antimeridian into two parts, so no single geometry spans the -180/+180 seam.

> [!question]- Q: What spatial data structures are particularly affected by antimeridian crossings?
> A: R-Trees and similar spatial indexes built on Cartesian assumptions fail to correctly index geometries that wrap around the antimeridian.


## Apache Gravitino

> [!question]- Q: What is Apache Gravitino?
> A: Apache Gravitino is an open-source unified metadata lake — a single layer that manages metadata across heterogeneous data sources and compute engines, rather than having separate catalogs per system.

> [!question]- Q: What is the key problem Gravitino solves?
> A: In complex data platforms with multiple engines (Spark, Trino, Flink, Hive) and multiple sources (relational DBs, data lakes, Iceberg tables), each system typically has its own catalog. Gravitino federates them all into one consistent metadata and governance layer.

> [!question]- Q: What is Gravitino comparable to?
> A: It is the open-source, vendor-neutral alternative to Databricks Unity Catalog — providing a single metadata control plane over a complex, multi-engine data platform.

> [!question]- Q: What are Gravitino's key capabilities?
> A: Multi-engine support (Spark, Trino, Flink, Hive), multi-source federation (relational DBs, data lakes, Iceberg tables), centralized governance and access control, and federated queries across sources through a consistent interface.


## Apache Hudi

> [!question]- Q: What does HUDI stand for?
> A: HUDI stands for Hadoop Upserts Deletes and Incrementals.

> [!question]- Q: Who created Apache Hudi and why?
> A: Uber built Hudi in 2016 (open-sourced 2019) to solve the problem of ingesting billions of trip events per day into their data lake with low latency while supporting record-level updates and deletes — requirements that batch-focused table formats like Iceberg and Delta did not address well.

> [!question]- Q: What are the two storage layouts in Apache Hudi?
> A: Copy-on-Write (CoW) — all data stored as Parquet files, rewrites files on every update, fast reads but expensive writes; and Merge-on-Read (MoR) — base Parquet files plus append-only Avro delta logs, very fast writes but reads merge base and delta on the fly, periodically compacted.

> [!question]- Q: When is Apache Hudi the right choice over Iceberg or Delta?
> A: Hudi excels when you have high-frequency record-level upserts from streaming sources (Kafka, Flink), need near-real-time data availability, or need efficient CDC (change data capture). Iceberg is better for multi-engine support; Delta is better on Databricks.

> [!question]- Q: What is Hudi's position in the "table format wars"?
> A: Hudi is the third major table format alongside Delta Lake and Apache Iceberg, though it is most common at companies with large-scale streaming pipelines (ride-sharing, e-commerce, ad-tech).


## Apache Iceberg

> [!question]- Q: What is Apache Iceberg?
> A: Apache Iceberg is an open table format for large analytic datasets, originally from Netflix (2017). It sits on top of Parquet files in object storage (S3, GCS, etc.) and adds database-like capabilities to raw files.

> [!question]- Q: What core problems does Iceberg solve for Parquet files in object storage?
> A: Raw Parquet files in S3 require scanning everything, cannot do atomic updates, and suffer from concurrent writer conflicts. Iceberg adds schema evolution, time travel, ACID transactions, partition evolution, and hidden partitioning.

> [!question]- Q: How does Iceberg work architecturally?
> A: Iceberg maintains a metadata layer — a catalog and manifest files — that tracks exactly which data files make up the current and historical table state. The actual data remains in Parquet/ORC/Avro; Iceberg is the bookkeeping layer on top.

> [!question]- Q: What query engines work with Apache Iceberg?
> A: Iceberg is typically paired with Apache Spark, Trino, and DuckDB. It is used at scale at Netflix, Apple, and LinkedIn.

> [!question]- Q: What is "time travel" in the context of Iceberg?
> A: Time travel allows you to query the table as it looked at any past snapshot, because Iceberg tracks the complete history of table state through its metadata layer.


## Apache Parquet

> [!question]- Q: What is Apache Parquet?
> A: Apache Parquet is a columnar binary file format for large tabular datasets, used internally by data warehouses like Google BigQuery, Snowflake, and DuckDB.

> [!question]- Q: Why is columnar storage advantageous in Parquet?
> A: Columnar storage allows computing statistics on a single column across millions of rows without reading other columns. It also enables built-in compression (Snappy or Zstandard) and predicate pushdown, where query engines skip row groups that don't match a filter without decompressing them.

> [!question]- Q: What is GeoParquet?
> A: GeoParquet is an open-standard spatial extension to Apache Parquet that adds geospatial data types (points, lines, polygons) to the Parquet format.

> [!question]- Q: What is predicate pushdown in Parquet?
> A: Predicate pushdown allows query engines to skip row groups that don't match a filter condition without decompressing or reading those groups, significantly reducing I/O for filtered queries.


## Apache Sedona

> [!question]- Q: What is Apache Sedona?
> A: Apache Sedona (formerly GeoSpark) is an open-source cluster computing system that extends Apache Spark and Apache Flink with native spatial datatypes, geospatial indexes, and Spatial SQL, enabling distributed geospatial analysis at scales single-machine tools cannot handle.

> [!question]- Q: What problem does Sedona solve that GeoPandas and PostGIS cannot?
> A: GeoPandas is single-threaded and in-memory, hitting limits at tens of millions of geometries. PostGIS is single-node with limited parallelism. Sedona handles billions of GPS points, global building footprints, or continental-scale satellite polygons through distributed compute.

> [!question]- Q: What are the three optimized spatial join strategies Sedona implements?
> A: (1) Partition-based spatial merge join (partitions both datasets by space so only nearby geometries are compared), (2) broadcast join (for small datasets, broadcasts the smaller one to every executor), and (3) index-accelerated join (builds a local R-Tree on one side within each partition).

> [!question]- Q: Who created Sedona and what is SedonaDB?
> A: Sedona was created by Jia Yu and Mohamed Sarwat (who later founded Wherobots). SedonaDB is Wherobots' commercial managed cloud service for Sedona — analogous to Databricks for Spark.

> [!question]- Q: What SQL standard does Sedona implement for spatial functions?
> A: Sedona implements the OGC Simple Features spatial function set, compatible with PostGIS syntax (e.g., `ST_Point(...)`, `ST_Contains(...)`).


## Archive

> [!question]- Q: What is an archive format?
> A: An archive format is a file format that stores one or more other files, optionally with compression applied. Examples include ZIP archives and PMTiles.

> [!question]- Q: How does PMTiles function as an archive format?
> A: PMTiles is a single-file archive that stores all map tiles (raster or vector) in one file with a spatial index, enabling cloud-optimized access to individual tiles via HTTP range requests without a tile server.


## Area of Interest

> [!question]- Q: What does AOI stand for?
> A: AOI stands for Area of Interest.

> [!question]- Q: What is an Area of Interest in geospatial contexts?
> A: An Area of Interest (AOI) is a geographic region defined by a user or system that bounds the spatial scope of a query, analysis, or tasking request. It is typically expressed as a bounding box or polygon and used to filter satellite imagery catalogs, task sensors, or constrain processing.


## Attitude

> [!question]- Q: What is "attitude" in the context of satellites?
> A: Attitude refers to the orientation of the satellite in space — specifically roll, pitch, and yaw — describing which way it is pointing, not its position.

> [!question]- Q: What hardware components measure and control satellite attitude?
> A: Attitude is measured by star trackers (identify star patterns for precise orientation), sun sensors (coarser reference), and IMUs (inertial measurement units). It is controlled by reaction wheels (spinning flywheels that transfer angular momentum), magnetorquers (use Earth's magnetic field, slow but low-power), and thrusters (propellant-based, for larger maneuvers).

> [!question]- Q: How do reaction wheels control satellite attitude?
> A: Reaction wheels are spinning flywheels inside the satellite. Changing their spin rate transfers angular momentum to the satellite body, rotating it without consuming propellant — purely electrically powered.


## Attitude Determination and Control System

> [!question]- Q: What does ADCS stand for?
> A: ADCS stands for Attitude Determination and Control System.

> [!question]- Q: What does an ADCS do on a satellite?
> A: The ADCS measures and controls the satellite's orientation (attitude) in space. It uses sensors (star trackers, sun sensors, magnetometers, IMUs) to determine the current orientation and actuators (reaction wheels, magnetorquers, thrusters) to adjust it to the desired pointing direction.

> [!question]- Q: Why is ADCS critical for Earth Observation satellites?
> A: EO satellites must precisely point their sensors at a target location during an acquisition, and may need to slew (rotate) to off-nadir angles to image targets that aren't directly below. Without a functioning ADCS, accurate, targeted imaging is impossible.


## Automatic Identification System

> [!question]- Q: What does AIS stand for and what is it?
> A: AIS stands for Automatic Identification System. It is an automated, autonomous tracking system used on ships to prevent collision and aid navigation, operating via VHF radio.

> [!question]- Q: What information does an AIS broadcast?
> A: AIS broadcasts a vessel's ID (MMSI number, ship name, IMO number), position, course, speed, and cargo type to both other ships and shore stations.

> [!question]- Q: What vessels are required to carry AIS?
> A: AIS is mandatory (via the International Maritime Organization) for most ships over 300 GT (Gross Tonnage).

> [!question]- Q: What is Satellite AIS (S-AIS) and why is it needed?
> A: VHF radio is limited to line-of-sight (~80 km), so terrestrial AIS cannot track ships in the open ocean. Satellite AIS (S-AIS) receives AIS signals from space, enabling global vessel tracking far from shore.

> [!question]- Q: What are the vulnerabilities of AIS?
> A: Not all vessels carry AIS, it can be switched off deliberately, and it is susceptible to spoofing, jamming, and signal interference in high-traffic areas.


## Backscatter

> [!question]- Q: What is backscatter in the context of SAR remote sensing?
> A: Backscatter is the portion of radar energy transmitted by a SAR sensor that is reflected back toward the sensor from the Earth's surface. The intensity and characteristics of backscatter reveal surface properties such as roughness, moisture content, and structure.

> [!question]- Q: What factors influence SAR backscatter?
> A: Surface roughness (rough surfaces scatter more energy back), dielectric properties (water content affects radar reflectivity), vegetation structure, incidence angle, and radar wavelength (band) all influence backscatter strength and pattern.

> [!question]- Q: How does incidence angle affect backscatter interpretation?
> A: At steeper incidence angles, backscatter is generally higher for smooth surfaces; at shallower angles, volume scattering from vegetation canopies becomes more prominent. Incidence angle must be accounted for when comparing acquisitions.


## Basemap

> [!question]- Q: What is a basemap?
> A: A basemap is a pre-rendered or pre-styled background map that provides geographic context (roads, place names, terrain, water bodies, etc.) over which user data layers are displayed.

> [!question]- Q: What is the difference between raster tile and vector tile basemaps?
> A: Raster tile basemaps serve pre-rendered PNG/JPEG images at each zoom level — simple and universally supported but with fixed styling. Vector tile basemaps serve raw vector geometries (in formats like Mapbox Vector Tile) styled client-side in WebGL, enabling dynamic restyling, layer hiding, and 3D extrusions without server re-rendering.

> [!question]- Q: Name some free and commercial basemap providers.
> A: Free: OpenStreetMap, Stadia Maps, Protomaps (PMTiles), MapTiler. Commercial: Mapbox, Google Maps, Apple Maps, ESRI/ArcGIS, HERE Technologies.

> [!question]- Q: What are common satellite/aerial imagery basemap sources?
> A: ESRI World Imagery (widely used free satellite basemap), Mapbox Satellite (Maxar imagery in a global mosaic), Google Satellite, and Bing Aerial.


## BigEarthNet

> [!question]- Q: What is BigEarthNet?
> A: BigEarthNet is a large-scale multi-label Sentinel-2 benchmark archive for remote sensing machine learning, published in February 2019. It contains 590,326 Sentinel-2 image patches annotated with multiple land-cover classes from the CORINE Land Cover 2018 database.

> [!question]- Q: What are the patch dimensions in BigEarthNet?
> A: Each image patch is 120×120 pixels for 10m bands, 60×60 pixels for 20m bands, and 20×20 pixels for 60m bands.

> [!question]- Q: What makes BigEarthNet distinctive as a benchmark?
> A: Unlike most existing archives, each patch is annotated with multiple land-cover classes (multi-label, not single-label). It is also significantly larger than prior remote sensing archives, making it well-suited as a training source for deep learning models.

> [!question]- Q: What was demonstrated in the BigEarthNet paper about CNN training?
> A: A shallow CNN trained on BigEarthNet achieved much higher accuracy for scene classification than a state-of-the-art CNN pre-trained on ImageNet, demonstrating the value of domain-specific large-scale remote sensing datasets.


## Bing QuadKey

> [!question]- Q: What is a Bing QuadKey?
> A: A Bing QuadKey is Microsoft's tile addressing system for Bing Maps, based on a QuadTree. Each tile is identified by a string of digits (0–3) that traces the path down the quadtree, with the string length equal to the zoom level.

> [!question]- Q: How do QuadKey digits encode spatial position?
> A: Each digit represents which quadrant of the parent tile the child occupies: 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right. For example, tile "213" is the bottom-left child of the top-right child of the bottom-right root quadrant.

> [!question]- Q: Is Bing QuadKey a Discrete Global Grid System (DGGS)?
> A: No. QuadKey is a tile addressing scheme on a flat Mercator projection, not a spherical partition. It inherits Mercator distortions and is more comparable to XYZ slippy map coordinates. It is designed for map display, not spatial data analysis, unlike H3 or S2.


## BlackSky

> [!question]- Q: What is BlackSky?
> A: BlackSky is a geospatial intelligence company operating a constellation of small (~55kg) LEO optical imaging satellites, providing high-frequency Earth observation imagery and integrated AI/ML analytics. It competes with Planet and Maxar but differentiates on high-revisit tasking and automated intelligence products.

> [!question]- Q: What is BlackSky's "Spectra" platform?
> A: Spectra is BlackSky's analytics platform that provides automated monitoring, change detection, and event alerting on top of its satellite imagery stream.

> [!question]- Q: Who are BlackSky's primary customers?
> A: Primarily government customers (defense and intelligence agencies) and commercial clients requiring persistent area monitoring.


## Building Information Modeling

> [!question]- Q: What does BIM stand for and what is it?
> A: BIM stands for Building Information Modeling. It is both a process and a data format for representing buildings and infrastructure as intelligent 3D models where every element carries structured data — not just geometry, but metadata about what the element is, its material, fire rating, thermal properties, relationships to other elements, cost, and manufacturer.

> [!question]- Q: What is IFC and how does it relate to BIM?
> A: IFC (Industry Foundation Classes) is the open standard file format for BIM data, maintained by buildingSMART International. It defines a schema for representing buildings with hundreds of typed entity classes and is the standard interchange format between BIM software.

> [!question]- Q: What is the challenge of using BIM models in real-time 3D visualization?
> A: BIM models are designed for construction workflows, not real-time rendering. They are often enormous (GBs), unoptimized for rendering (thousands of small objects, complex geometries), and poorly georeferenced, making them difficult to stream as 3D Tiles into platforms like Cesium.

> [!question]- Q: What is the connection between BIM and geospatial digital twins?
> A: A BIM model can be georeferenced, converted to 3D Tiles, and streamed into Cesium, then overlaid with real-time IoT sensor data (occupancy sensors, energy meters, HVAC state). The BIM geometry provides spatial context while IoT data provides real-time state.

> [!question]- Q: Name the major BIM software packages.
> A: Autodesk Revit (dominant in architecture/construction), ArchiCAD (strong in Europe), Bentley OpenBuildings (infrastructure), Tekla Structures (structural steel/concrete), and Autodesk Navisworks (clash detection and coordination).


## Canadian Space Agency

> [!question]- Q: What does CSA stand for?
> A: CSA stands for the Canadian Space Agency (Agence spatiale canadienne in French).

> [!question]- Q: What is the Canadian Space Agency known for in Earth Observation?
> A: The CSA is best known for the RADARSAT program — a series of SAR (Synthetic Aperture Radar) satellites. RADARSAT-2 and the RADARSAT Constellation Mission (RCM, three satellites launched 2019) provide C-Band SAR coverage with applications in ice monitoring, maritime surveillance, disaster response, and land use.

> [!question]- Q: What is the RADARSAT Constellation Mission (RCM)?
> A: The RCM is a constellation of three C-Band SAR satellites operated by the CSA, launched in 2019, designed to provide frequent revisits for applications such as maritime monitoring, agriculture, and emergency management across Canada and globally.


## Canopy Height Model

> [!question]- Q: What does CHM stand for and what is it?
> A: CHM stands for Canopy Height Model. It is a raster dataset representing the height of vegetation above the ground surface — essentially a map of how tall trees and other vegetation are at every point in a landscape.

> [!question]- Q: How is a Canopy Height Model derived?
> A: CHM = DSM (Digital Surface Model, elevation of the top of everything) minus DTM (Digital Terrain Model, bare-earth elevation). The difference isolates the height of vegetation and structures above the ground.

> [!question]- Q: What data sources are used to generate CHMs?
> A: LiDAR (the gold standard), photogrammetry (Structure from Motion), satellite stereo imagery, and spaceborne LiDAR such as NASA's GEDI (Global Ecosystem Dynamics Investigation).


## Capella Space

> [!question]- Q: What does Capella Space specialize in?
> A: Capella Space specializes in commercial X-Band SAR (Synthetic Aperture Radar) satellites, providing very high resolution radar imagery capable of imaging through clouds and at night.

> [!question]- Q: What SAR band does Capella use and what are its tradeoffs?
> A: Capella uses X-Band — the finest/smallest wavelength, giving the highest detail and manageable antenna size, excellent for urban areas, infrastructure, and vehicles. The tradeoff is less penetration depth: it cannot see through forest canopy or soil like L-Band (e.g., PALSAR), making it less useful for biomass or soil moisture applications.

> [!question]- Q: What is Capella's current operational constellation?
> A: The Acadia constellation (2023), consisting of approximately 7 satellites with 0.25m resolution. An earlier Sequoia constellation (Gen 2, ~8 satellites, 0.5m resolution) has since decayed.

> [!question]- Q: How does Capella compare to ICEYE?
> A: Both operate commercial SAR constellations, but ICEYE operates the world's largest commercial SAR fleet, all in X-Band. Capella differentiates on higher resolution (0.25m for Acadia) and focuses on intelligence and analytics use cases.


## Cesium

> [!question]- Q: What is CesiumJS?
> A: CesiumJS is a WebGL-based open-source library for 3D globe and map visualization in the browser. Unlike 2D mapping libraries (MapLibre, Leaflet), Cesium is genuinely 3D from the ground up — Earth is a sphere, objects exist in 3D space, and you can fly through cities or view from space.

> [!question]- Q: What is CZML?
> A: CZML is Cesium's own JSON format for describing time-dynamic scenes — things that move over time, such as satellites along orbits, vehicles following paths, or sensors sweeping areas.

> [!question]- Q: What is Cesium Ion?
> A: Cesium Ion is Cesium's commercial cloud platform that provides hosted terrain (Cesium World Terrain), hosted imagery (Bing Maps, Sentinel-2 mosaic), asset hosting, and a tiling pipeline that converts raw data (LAS, OBJ, GeoTIFF) into streamable 3D Tiles.

> [!question]- Q: What types of data can CesiumJS render?
> A: Terrain (Quantized Mesh format), imagery (XYZ tile layers draped over terrain), 3D Tiles (photogrammetry meshes, LiDAR point clouds, BIM data, CityGML), and arbitrary 3D glTF models placed at real-world coordinates.

> [!question]- Q: What is the relationship between Cesium and 3D Tiles?
> A: Cesium invented the 3D Tiles specification, which has since become an OGC standard. CesiumJS remains the primary consumer, but 3D Tiles is now supported by other tools including deck.gl, QGIS, Unreal Engine, Unity, and ArcGIS.


## Clay

> [!question]- Q: What is Clay in the context of geospatial AI?
> A: Clay is an open-weight geospatial foundation model released in May 2024, built by the Clay Foundation (within Renaissance Philanthropy). It uses a Vision Transformer architecture trained as a self-supervised Masked Autoencoder (MAE) on multi-spectral, multi-temporal satellite imagery (Sentinel-2, Landsat, NAIP).

> [!question]- Q: What are the primary use cases for Clay?
> A: Generating semantic embeddings for any location and time (for feature detection, classification, regression), fine-tuning for downstream tasks (land cover classification, above-ground biomass prediction, change detection), and using as a backbone for other models.

> [!question]- Q: How does Clay differ from AlphaEarth Foundations?
> A: Clay is open-weight (can be fine-tuned), while AlphaEarth Foundations is not. Clay was trained primarily on optical imagery (Sentinel-2, Landsat, NAIP), while AEF fuses optical, radar, LiDAR, climate, and elevation data. AEF outperforms Clay on benchmark evaluations.


## Cloud-Native Geospatial

> [!question]- Q: What is Cloud-Native Geospatial?
> A: Cloud-Native Geospatial is a philosophy, set of formats, and architectural patterns for storing, accessing, and processing geospatial data designed from the ground up for cloud object storage — enabling compute to come to the data rather than downloading data to compute.

> [!question]- Q: What is the Cloud-Native Geospatial Foundation (CNG)?
> A: CNG is a Linux Foundation project that stewards key open geospatial standards including STAC, COG, GeoParquet, COPC, and PMTiles. It brings together organizations (AWS, Microsoft, Google, ESRI, Planet, Maxar) to coordinate standard development.

> [!question]- Q: What are the four core principles of Cloud-Native Geospatial?
> A: (1) Data stays where it is (compute comes to data via HTTP range requests), (2) formats optimized for partial access (spatial indexes/headers enabling targeted byte-range reads), (3) serverless where possible (no tile or processing servers needed), and (4) separation of storage and compute (same data can be read by Python, a browser, QGIS, or a Spark cluster).

> [!question]- Q: What is the Cloud-Native Geospatial stack for different data types?
> A: Vector: GeoParquet (analytics), FlatGeobuf (streaming/queries), PMTiles (map display). Raster: COG (imagery/scenes), Zarr (multidimensional/time series). Point clouds: COPC (single-file), EPT (tiled). Catalogs: STAC.

> [!question]- Q: What underlying HTTP mechanism enables cloud-native geospatial access?
> A: HTTP Range Requests — the ability to request specific byte ranges of a file stored in object storage (S3, GCS, R2) without downloading the entire file.


## Cloud-Optimized GeoTIFF

> [!question]- Q: What does COG stand for and what is it?
> A: COG stands for Cloud-Optimized GeoTIFF. It is a variant of the GeoTIFF format with a specific internal layout that enables efficient partial access over a network via HTTP range requests. All COG files are valid GeoTIFFs, but not all GeoTIFFs are COGs.

> [!question]- Q: What are the two key internal features that make a COG cloud-optimized?
> A: (1) Internal tiling — the image is divided into spatial tiles (e.g., 256×256 or 512×512 pixels) so a single tile can be read without touching the rest of the file. (2) Overview levels (pyramids) — lower-resolution versions of the image are embedded in the same file, so a reader can pick the appropriate resolution level for the current zoom without fetching full-resolution data.

> [!question]- Q: How does a COG reader efficiently serve a specific zoom level view?
> A: The reader first fetches the header to find the offset of the needed overview level, then sends targeted HTTP range requests for only the specific tiles within that overview level that intersect the viewport — typically just two HTTP requests total.

> [!question]- Q: What is the recommended tile size for COG files?
> A: 256×256 or 512×512 pixel dimensions are recommended. Too-large single files increase header size and may require multiple requests just to read the spatial index; too-small files create excessive read overhead for combined visualizations.

> [!question]- Q: What tool commonly consumes COGs to serve map tiles?
> A: TiTiler is a commonly used tile server that consumes COGs directly from object storage to serve XYZ map tiles on demand.


## Cloud-Optimized Point Cloud

> [!question]- Q: What does COPC stand for and what is it?
> A: COPC stands for Cloud-Optimized Point Cloud. It is a cloud-optimized variant of the LAZ (compressed LAS) file format for LiDAR point clouds. A COPC file reorganizes points into a spatially-indexed structure using a spatial octree, enabling partial reads via HTTP range requests.

> [!question]- Q: How does COPC relate to LAZ and LAS?
> A: LAS is the standard format for 3D point cloud data from LiDAR. LAZ is a compressed LAS file. COPC is a cloud-optimized LAZ file — one LAS/LAZ file becomes one COPC file with the same data reorganized for partial access.

> [!question]- Q: How does COPC relate to COG for raster data?
> A: COPC is to LAZ as COG is to GeoTIFF: both are cloud-optimized versions of an existing format that add spatial indexing (an octree for COPC, internal tiles + overviews for COG) to enable efficient HTTP range request access.

> [!question]- Q: When would you use COPC vs. Entwine Point Tiles (EPT)?
> A: COPC is for single-file datasets (one LAS/LAZ becomes one COPC). EPT is for multi-file datasets (e.g., a LiDAR survey across hundreds of LAZ files). The relationship is roughly: COPC:EPT :: COG:Zarr.


## CloudCompare

> [!question]- Q: What is CloudCompare?
> A: CloudCompare is an open-source desktop application for point cloud and mesh visualization, analysis, and processing. It is the primary tool for interactively inspecting and analyzing point clouds.

> [!question]- Q: What are CloudCompare's main use cases?
> A: Visualizing and inspecting point clouds interactively, manual classification editing, distance computation between two point clouds (M3C2 algorithm for change detection and erosion monitoring), ICP (Iterative Closest Point) registration for aligning point clouds, and roughness/curvature/noise estimation.

> [!question]- Q: What is CloudCompare's role in a geospatial workflow?
> A: CloudCompare is an interactive analysis and quality-checking environment, not a pipeline tool. It is essential for ground-truthing automated results and performing one-off analyses or visual inspections.


## Comma-Separated Values

> [!question]- Q: What does CSV stand for?
> A: CSV stands for Comma-Separated Values.

> [!question]- Q: What is a CSV file and what are its limitations for geospatial use?
> A: A CSV is a plain-text tabular format where values are separated by commas and rows by newlines. For geospatial data, CSVs can store point coordinates (latitude/longitude columns) but have no native support for complex geometries (polygons, lines), no standardized CRS metadata, no binary compression, and poor performance for large datasets compared to formats like GeoParquet or GeoJSON.

> [!question]- Q: When is CSV appropriate for geospatial data?
> A: CSV is appropriate for small point datasets, interoperability with non-GIS tools (spreadsheets, databases), and human-readable data exchange where simplicity matters more than performance or geometric richness.


## Compression

> [!question]- Q: What is compression and what is the core tradeoff?
> A: Compression is an algorithm that makes data smaller by encoding it into a compact representation. The tradeoff is encoding and decoding time, but in most cases the benefit of smaller stored file sizes outweighs this cost.

> [!question]- Q: What is the difference between lossy and lossless compression?
> A: Lossy compression permanently discards information — exact original values cannot be recovered (examples: JPEG, LERC). Lossless compression allows exact recovery of the original values (examples: gzip, ZSTD, LZW, deflate). Lossless files are typically larger than lossy-compressed equivalents.

> [!question]- Q: What is the difference between external and internal compression?
> A: External compression is applied after a file is saved (e.g., wrapping in a ZIP or gzip), and typically makes the file no longer cloud-optimized since byte ranges can't map to internal spatial chunks. Internal compression is part of the file format's specification (e.g., COG, COPC, GeoParquet), enabling cloud-optimized access where internal chunks can still be fetched individually via HTTP range requests.

> [!question]- Q: Why should you not apply external gzip compression on top of a file that already uses internal compression?
> A: For formats that already use internal compression (COG, COPC, GeoParquet), adding external gzip will not meaningfully reduce file size and will add an extra decompression step, reducing read performance.


## Constellation

> [!question]- Q: What is a satellite constellation?
> A: A satellite constellation is a coordinated network of satellites working together as a system to provide continuous coverage of an area or the whole Earth. Coverage continuity is achieved by having enough satellites so that as one passes out of view, another appears.

> [!question]- Q: What is a Walker Constellation?
> A: A Walker Constellation is a specific mathematical pattern for arranging satellites across multiple orbital planes to achieve uniform, predictable coverage, defined by T/P/F: total number of satellites, number of orbital planes, and phase factor (offset between adjacent planes). GPS is a Walker 24/6/2 constellation.

> [!question]- Q: What are the two Walker Constellation variants?
> A: Walker Star: near-polar orbital planes distributed like longitude lines, meeting at poles — good for polar coverage (used by Iridium). Walker Delta: inclined but non-polar planes, distributed symmetrically — better for mid-latitude coverage (used by GPS).

> [!question]- Q: Why do lower-orbit constellations require more satellites?
> A: Lower orbit means a smaller footprint per satellite. GPS achieves global coverage with ~24 satellites at 20,200km altitude, while Starlink needs 4,000+ satellites at 550km to provide equivalent global internet coverage.

> [!question]- Q: What is revisit rate and why does it matter for Earth Observation constellations?
> A: Revisit rate is how often at least one satellite passes over a given point. Planet's Dove constellation achieves daily global revisits, while a single Landsat satellite revisits every 16 days. Higher revisit rate enables more frequent change monitoring.


## Continuous Change Detection and Classification

> [!question]- Q: What does CCDC stand for?
> A: CCDC stands for Continuous Change Detection and Classification.

> [!question]- Q: How does the CCDC algorithm work?
> A: CCDC fits harmonic (sinusoidal) models to a pixel's full spectral time series (typically decades of Landsat observations) for each spectral band. It then monitors residuals: when observations deviate significantly from the model, a change is detected. After a change, a new harmonic model is fit to the new stable period.

> [!question]- Q: What outputs does CCDC produce per pixel?
> A: Harmonic coefficients (amplitude, phase, RMSE) for each spectral band, change dates and magnitudes, and land cover classification at any point in time.

> [!question]- Q: What are CCDC's strengths and weaknesses?
> A: Strengths: explicitly captures seasonal phenology, detects abrupt changes (deforestation, fire, urban conversion) precisely in time, and produces dense temporal features even from sparse cloudy observations. Weaknesses: designed around Landsat cadence specifically, purely spectral with no spatial context, and hand-engineered rather than learned.

> [!question]- Q: What satellite data is CCDC primarily designed for?
> A: CCDC is designed around Landsat time series, leveraging decades of consistent multispectral observations at 30m resolution with 16-day revisit cadence.
## Copernicus DEM

> [!question]- Q: What is the Copernicus DEM and what are its two main variants?
> A: A free, global Digital Elevation Model derived from Airbus's WorldDEM dataset (itself from TanDEM-X radar interferometry). The two variants are GLO-30 (30m resolution) and GLO-90 (90m resolution).

> [!question]- Q: Why is the Copernicus DEM preferred over SRTM for most new geospatial work?
> A: It is more recent, has better accuracy, fewer voids, handles vegetated areas and steep terrain better, and has been hydrologically conditioned to remove radar artifacts over water bodies.

> [!question]- Q: What makes the Copernicus DEM "edited" compared to raw TanDEM-X data?
> A: The raw interferometric data has artifacts over water bodies (radar phase is noisy over water), voids over steep terrain, and other issues. Copernicus DEM is a processed, cleaned, and hydrologically conditioned version, resulting in cleaner and more usable data.

> [!question]- Q: What is the highest-resolution freely available global DEM as of 2026?
> A: The Copernicus GLO-30 at 30m resolution.

> [!question]- Q: Where can the Copernicus DEM be accessed?
> A: It is available on the AWS Open Data Registry, Copernicus Data Space Ecosystem, and OpenTopography.


## Dagster

> [!question]- Q: What is Dagster?
> A: Dagster is a data orchestration platform for building, scheduling, and monitoring data pipelines. It is an alternative to tools like Apache Airflow and Prefect, with a strong emphasis on software-defined assets and data quality.

> [!question]- Q: How does Dagster differ from Apache Airflow in its core abstraction?
> A: Airflow is task-centric (you define tasks and their dependencies), while Dagster is asset-centric (you define data assets and how they are produced), making it easier to reason about data lineage and freshness.

> [!question]- Q: What problem does Dagster solve?
> A: It solves the problem of orchestrating complex data pipelines with observability, testability, and data quality checks built in, reducing the operational overhead of managing dependencies between data transformation steps.


## Data Lakehouse

> [!question]- Q: What is a Data Lakehouse?
> A: An architectural pattern that combines the low-cost, flexible storage of a Data Lake with the structure, ACID transactions, and query performance of a data warehouse in a single system.

> [!question]- Q: What problem does the Data Lakehouse pattern solve?
> A: Organizations traditionally had to build and maintain both a data lake (for cheap storage and ML) and a data warehouse (for analytics and BI), duplicating data and adding latency. The Lakehouse collapses both into one system.

> [!question]- Q: What technology enables the Lakehouse pattern?
> A: Open table formats (Delta Lake, Apache Iceberg, Apache Hudi) — a metadata and transaction layer that sits on top of Parquet files in object storage, giving them warehouse-like properties such as ACID transactions, time travel, and schema evolution.

> [!question]- Q: Name four major Data Lakehouse implementations.
> A: Databricks Lakehouse (built on Delta Lake), Apache Iceberg + open stack (Spark/Trino + Polaris), AWS Lake Formation (S3 + Glue + Athena), and Google BigLake (BigQuery extended to GCS).

> [!question]- Q: How does the Lakehouse pattern apply to geospatial data?
> A: GeoParquet serves as the file format, Apache Iceberg or Delta Lake as the table format, DuckDB Spatial or Apache Sedona as the query engine, and STAC as the catalog for raster/imagery assets alongside the vector/tabular lakehouse.


## Databricks Unity Catalog

> [!question]- Q: What is Databricks Unity Catalog?
> A: Databricks's unified governance and metadata management layer for data and AI assets in the Databricks Lakehouse platform, providing centralized access control, auditing, lineage tracking, and discovery.

> [!question]- Q: What types of assets does Unity Catalog govern?
> A: Data tables, ML models, notebooks, dashboards, and feature store assets — all within a single governance framework.

> [!question]- Q: What is the alias for Databricks Unity Catalog?
> A: Unity Catalog.


## dbt

> [!question]- Q: What does dbt stand for and what does it do?
> A: dbt stands for Data Build Tool. It is an open-source transformation framework that allows data analysts and engineers to write SQL SELECT statements and have dbt handle materializing them as tables or views, manage dependencies, test data quality, and generate documentation.

> [!question]- Q: What is the fundamental unit in dbt and what does it correspond to?
> A: A model — a SQL file containing a SELECT statement. Each model corresponds to a table or view in your data warehouse.

> [!question]- Q: What are the five materialization types in dbt?
> A: view (SQL view, no stored data), table (full table recreated on every run), incremental (only new/changed records processed on subsequent runs), ephemeral (injected as a CTE, never materialized), and materializedview (uses the warehouse's native materialized view feature).

> [!question]- Q: How does dbt manage dependencies between models?
> A: Through the ref() function. dbt parses all ref() calls across models, builds a DAG of dependencies, and automatically executes models in the correct order, parallelizing where possible.

> [!question]- Q: Where does dbt fit in the modern data stack?
> A: dbt is the T (or L+T in ELT) layer: data is ingested by tools like Airbyte/Fivetran into raw tables, dbt transforms those into bronze/silver/gold layers, and BI tools or ML pipelines consume the output.


## Dead Letter Queue

> [!question]- Q: What is a Dead Letter Queue (DLQ)?
> A: A Dead Letter Queue is a message queue that stores messages that could not be successfully processed by a consumer after a configured number of retries, preventing them from blocking the main queue and allowing for later inspection and reprocessing.

> [!question]- Q: What problem does a Dead Letter Queue solve?
> A: It prevents "poison pill" messages — malformed or unprocessable messages — from repeatedly blocking a queue and causing processing to stall. Failed messages are routed to the DLQ for debugging and manual remediation.

> [!question]- Q: Where are Dead Letter Queues commonly used?
> A: In message broker systems like AWS SQS, Apache Kafka, RabbitMQ, and Azure Service Bus as part of fault-tolerant event-driven architectures.


## Delta Lake

> [!question]- Q: What is Delta Lake and who created it?
> A: Delta Lake is an open-source storage layer that brings ACID transactions, versioning, and schema management to data stored as Parquet files on object storage. It was originally developed by Databricks and open-sourced in 2019; it is now a Linux Foundation project.

> [!question]- Q: What core problem does Delta Lake solve?
> A: Object storage is cheap but fundamentally transactionless — a failed write mid-way through produces a partially corrupted table. Delta Lake adds a transaction log on top of plain Parquet files so that writes are atomic.

> [!question]- Q: What capabilities does the Delta Lake transaction log enable?
> A: ACID transactions, time travel (query previous table versions), schema evolution, schema enforcement, and upserts and deletes.

> [!question]- Q: How does Delta Lake reconstruct the current state of a table?
> A: By replaying the transaction log. The log is the source of truth; the Parquet files are just the data storage layer.


## Differential Interferometric Synthetic Aperture Radar

> [!question]- Q: What does DInSAR stand for?
> A: Differential Interferometric Synthetic Aperture Radar.

> [!question]- Q: How does DInSAR differ from InSAR?
> A: InSAR uses two or more radar images to measure surface topography and deformation. DInSAR is a specialized InSAR technique that removes the topographic phase component to isolate and measure precise surface displacements (millimeter-level) between acquisitions. InSAR measures elevation; DInSAR monitors motion.

> [!question]- Q: What is the primary application of DInSAR?
> A: Monitoring surface deformation — such as ground subsidence, landslides, volcanic inflation/deflation, and earthquake displacement — at millimeter precision.


## Digital Elevation Model

> [!question]- Q: What is a Digital Elevation Model (DEM)?
> A: A DEM is a generic term for any elevation raster — a gridded representation of terrain elevation values. It encompasses both Digital Surface Models (DSMs) and Digital Terrain Models (DTMs).

> [!question]- Q: What is the difference between a DSM and a DTM?
> A: A DSM (Digital Surface Model) represents the elevation of the top surface including buildings, trees, and all above-ground features. A DTM (Digital Terrain Model) represents bare-earth elevation with buildings and vegetation removed.

> [!question]- Q: How are DSMs and DTMs typically derived from LiDAR data?
> A: Raw LiDAR gives a DSM first (all returns). Ground points are then classified from non-ground returns via point cloud classification to produce a DTM. The difference between DSM and DTM at any pixel equals the height of features (buildings, trees) on the ground.


## Digital Surface Model

> [!question]- Q: What is a Digital Surface Model (DSM)?
> A: A DSM is a raster of the elevation of the top surface — it includes buildings, trees, powerlines, and everything above bare earth. It is a type of DEM of the surface.

> [!question]- Q: How is a DSM produced from raw LiDAR or photogrammetry data?
> A: Raw LiDAR or photogrammetric processing yields a DSM first as the initial product. The DSM must be further processed (classifying ground vs. non-ground returns) to derive a DTM (bare-earth model).


## Digital Terrain Model

> [!question]- Q: What is a Digital Terrain Model (DTM)?
> A: A DTM is a raster of bare-earth elevation with buildings and vegetation removed. It is a type of DEM representing only the terrain surface.

> [!question]- Q: What is the relationship between a DSM and a DTM?
> A: The DTM is derived from the DSM by classifying point cloud returns into ground and non-ground, and keeping only ground points. The pixel-wise difference between DSM and DTM gives the height of above-ground objects (buildings, trees).


## Directed Acyclic Graph

> [!question]- Q: What is a Directed Acyclic Graph (DAG)?
> A: A DAG is a graph data structure consisting of nodes connected by directed edges, with no cycles (no path leads back to a starting node). It is widely used in data engineering to represent dependency relationships between tasks or data transformations.

> [!question]- Q: Where are DAGs used in data engineering?
> A: In pipeline orchestration tools like Apache Airflow, Prefect, and Dagster — tasks are nodes and dependencies are directed edges, ensuring tasks execute in the correct order. dbt also builds a DAG of SQL model dependencies via the ref() function.

> [!question]- Q: Why must a pipeline dependency graph be acyclic?
> A: Cycles would create circular dependencies where task A depends on B, which depends on A — making execution order impossible to determine and causing infinite loops.


## Discrete Global Grid System

> [!question]- Q: What is a Discrete Global Grid System (DGGS)?
> A: A DGGS is a system that partitions the entire Earth's surface into a hierarchy of discrete cells, each with a unique identifier, covering the globe without gaps or overlaps, with cells at the same level being roughly equal area.

> [!question]- Q: Name four major DGGS options and their cell shapes/creators.
> A: S2 Geometry (Google, square-ish cells on cube projection), H3 (Uber, hexagonal cells), A5 (deck.gl creator, pentagonal cells), and Geohash (older system, rectangular cells, base-32 string IDs).

> [!question]- Q: What is H3's key advantage over S2 for analytics use cases?
> A: H3 uses hexagonal cells where all 6 neighbors are equidistant from the center (uniform adjacency), making it ideal for movement analysis, proximity calculations, and aggregation. S2's square-ish cells don't have this property.

> [!question]- Q: Why is S2 preferred for backend infrastructure over H3?
> A: S2's power-of-4 subdivision means every level is a clean bit-prefix of the level below, so deriving a coarser cell from a fine-resolution cell is nearly free (just bit-shifting). This makes S2 ideal for systems that need to dynamically work across multiple resolutions.

> [!question]- Q: What is a common misconception about DGGS replacing spatial indexes?
> A: DGGSs do not replace true spatial indexes like R-trees. Replacing PostGIS with H3 cell IDs in a B-tree index enables fast point lookups and approximate proximity, but complex operations (polygon intersection, buffer analysis, routing) still require proper geometries and spatial indexes. DGGSs complement spatial indexes rather than replace them.


## Dolly

> [!question]- Q: What is Dolly 2.0 and who created it?
> A: Dolly 2.0 is an open-source instruction-following LLM released by Databricks in April 2023. It is a 12B parameter model based on the Pythia model family, fine-tuned on a human-generated instruction-following dataset.

> [!question]- Q: What makes Dolly 2.0's training dataset distinctive?
> A: It consists of 15,000 English instruction-following examples written entirely by Databricks employees — both instructions and answers are human-generated, making it commercially licensable. This contrasts with datasets like Alpaca where responses were generated by ChatGPT, which has unfavorable commercial terms.

> [!question]- Q: What 7 use cases does the Dolly dataset cover?
> A: Open QA, closed QA, information extraction, summarization of Wikipedia data, brainstorming, classification, and creative writing.

> [!question]- Q: What was the motivation for creating a new dataset for Dolly 2.0 rather than reusing Dolly 1.0's dataset?
> A: Dolly 1.0 used a dataset generated via the OpenAI API (the Alpaca dataset), whose terms of service prohibit using outputs to train competing models. A fully human-generated dataset was needed for legitimate commercial use.


## DuckDB

> [!question]- Q: What is DuckDB and how does it differ from traditional databases?
> A: DuckDB is an in-process analytical (OLAP) database — it runs embedded inside your application process (like SQLite) rather than as a separate server. It is free, open-source (MIT), and primarily written in C++.

> [!question]- Q: What execution model does DuckDB use and why?
> A: DuckDB uses vector-at-a-time (vectorized) execution, processing chunks of rows at a time sized to fit in CPU cache. This is the best of both worlds between row-at-a-time (low memory, high CPU overhead) and column-at-a-time (fast but large memory footprint).

> [!question]- Q: What does DuckDB's "graceful degradation" mean?
> A: Almost all DuckDB operators spill to disk when they run out of RAM, rather than crashing. The system aims to "never crash, always make progress."

> [!question]- Q: What does the DuckDB Spatial extension provide?
> A: A GEOMETRY type (points, lines, polygons, multipolygons) modeled after PostGIS, with no runtime dependencies — it statically bundles GDAL, GEOS, and PROJ, and embeds the PROJ projection database supporting over 3,000 CRS definitions.

> [!question]- Q: What is DuckDB not suitable for?
> A: DuckDB has no concurrent multi-client access, no persistent network connection pool, and no long-running server process. It is not appropriate for multi-process systems where multiple clients need simultaneous access to the same data.


## DuckDB Labs

> [!question]- Q: What is DuckDB Labs and who founded it?
> A: DuckDB Labs is the company behind DuckDB and DuckLake, founded by Hannes Mühleisen and Mark Raasveldt during their research at CWI (Dutch national research institute) in 2021.

> [!question]- Q: What is DuckDB Labs' business model given that DuckDB is MIT-licensed?
> A: DuckDB is MIT-licensed and will always remain free, but DuckDB Labs offers DuckDB Cloud (a managed cloud service), enterprise support contracts, and consulting and professional services.

> [!question]- Q: What has been DuckDB Labs' outsized impact on geospatial workflows?
> A: DuckDB's geospatial extension has made GeoParquet practically useful for ad-hoc geospatial analysis without requiring PostGIS, significantly lowering the barrier to analytical queries on large vector datasets.


## DuckLake

> [!question]- Q: What is DuckLake and who announced it?
> A: DuckLake is an open lakehouse catalog format announced by DuckDB Labs in early 2025, designed as a dramatically simpler alternative to existing catalog solutions like Apache Hive Metastore, AWS Glue, and Apache Polaris.

> [!question]- Q: What is DuckLake's core architectural innovation?
> A: The entire catalog (table metadata, snapshots, schema history, transaction log) lives inside a single DuckDB database file that can sit on S3, GCS, or local disk — no catalog service to run, just a file.

> [!question]- Q: What key lakehouse features does DuckLake support?
> A: Serverless operation, ACID transactions (inherited from DuckDB), time travel (snapshot tracking like Apache Iceberg), Apache Iceberg compatibility (can export tables as Iceberg metadata for Spark, Trino, Flink), and format-agnostic data files (Parquet or others).

> [!question]- Q: What problem with existing lakehouse catalogs does DuckLake solve?
> A: All existing catalogs (Hive Metastore, AWS Glue, Polaris, Gravitino, Project Nessie) require running infrastructure services. DuckLake requires zero infrastructure beyond a file on object storage.


## Earth Genome

> [!question]- Q: What is Earth Genome?
> A: Earth Genome is the organization behind the EarthIndex product, focused on applying AI and geospatial analysis to environmental and ecological data.

> [!question]- Q: What is EarthIndex?
> A: EarthIndex is a product by Earth Genome that enables search and analysis across large Earth observation datasets, applying AI to index and make planetary-scale environmental data discoverable.


## Electromagnetic Spectrum

> [!question]- Q: What is the electromagnetic spectrum?
> A: The range of all electromagnetic radiation, ordered by wavelength (or equivalently frequency). It spans from long-wavelength radio waves through microwave, infrared, visible light, ultraviolet, X-rays, and gamma rays (shortest wavelength, highest frequency).

> [!question]- Q: What is the relationship between wavelength and frequency in the EM spectrum?
> A: They are inversely related — a shorter wavelength means a higher frequency.

> [!question]- Q: Why does microwave energy have advantages for remote sensing?
> A: Microwave wavelengths can pass through clouds and atmospheric water vapor, allowing instruments using microwave energy (like SAR) to image the Earth's surface regardless of cloud cover.

> [!question]- Q: What is a spectral fingerprint in the context of remote sensing?
> A: Every material on Earth reflects, absorbs, or transmits EM energy in a unique pattern across wavelengths — its spectral signature. This unique fingerprint allows identification of materials (rock types, vegetation species, etc.) using multispectral or hyperspectral sensors.

> [!question]- Q: What is spectral resolution and why does it matter?
> A: Spectral resolution is the number and width of spectral bands a sensor captures. More bands enable finer differentiation between materials, since each material has a unique spectral fingerprint across wavelengths.


## Entwine Point Tiles

> [!question]- Q: What is EPT (Entwine Point Tiles) and who developed it?
> A: EPT is a cloud-friendly format for storing and serving large LiDAR point clouds, developed by Hobu Inc. (the same team behind PDAL). It organizes point cloud data as a hierarchical octree in a multi-file structure for efficient streaming and spatial querying.

> [!question]- Q: What problem does EPT solve?
> A: A national LiDAR dataset might have hundreds of billions of points. Without hierarchical structure, any query requires reading everything. EPT enables streaming in web viewers, efficient bounding-box clipping, and progressive loading (coarse overview first, refine as you zoom in).

> [!question]- Q: How is EPT structured spatially?
> A: As a hierarchical octree (3D analogue of a QuadTree). Root nodes contain subsampled points representing the whole dataset at coarse resolution; leaf nodes contain full-density data for small spatial regions. It is the 3D equivalent of a COG's overview pyramid.

> [!question]- Q: What file format do EPT nodes use to store point data?
> A: Each node is stored as a LAZ file containing the points that fall in that octree node at that level of detail.

> [!question]- Q: How does EPT compare to COPC (Cloud-Optimized Point Cloud)?
> A: Both serve cloud-native point cloud access. EPT uses a multi-file octree structure (many LAZ files). COPC is a single-file format following the COG philosophy — the entire octree is packed inside one LAZ file, which simplifies distribution. EPT is the current standard for national-scale distribution; COPC is newer and simpler to work with.


## Ephemeris

> [!question]- Q: What is a satellite ephemeris?
> A: A dataset providing the precise position and velocity (state vector) of a satellite at a specific time, allowing computation of where the satellite is or will be at any future time. The plural is ephemerides.

> [!question]- Q: What is a TLE (Two-Line Element Set)?
> A: A compact two-line ASCII format for encoding a satellite's orbital elements (inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion). It is the most common ephemeris format and is used with the SGP4 propagator algorithm to compute future satellite positions.

> [!question]- Q: What can be derived from a satellite ephemeris?
> A: Current and future satellite position, ground track and swath coverage, pass-over times for a target area of interest, and sun angle at acquisition time.

> [!question]- Q: Why do InSAR applications require precise ephemerides rather than broadcast ones?
> A: InSAR needs to isolate ground deformation from orbital error. Broadcast ephemerides are not accurate enough for this; precise ephemerides (post-processed, centimeter-level orbital data) are required.


## ETL

> [!question]- Q: What does ETL stand for?
> A: Extract, Transform, Load — a data integration pattern where data is extracted from source systems, transformed into the desired format/schema, and loaded into a target system such as a data warehouse.

> [!question]- Q: What is the difference between ETL and ELT?
> A: In ETL, data is transformed before loading into the destination (transformation happens outside the warehouse). In ELT (Extract, Load, Transform), raw data is loaded first into the destination, and transformation happens inside the warehouse using its compute power — the pattern favored by modern cloud warehouses and tools like dbt.

> [!question]- Q: What is Reverse ETL?
> A: Reverse ETL is the process of moving data from a data warehouse back to operational tools (CRM, ad platforms, customer success tools). Tools like Hightouch and Census implement this pattern.


## European Space Agency

> [!question]- Q: What is the European Space Agency (ESA)?
> A: ESA is the intergovernmental space agency of Europe, founded in 1975 and headquartered in Paris. It coordinates European space exploration, Earth observation, and satellite programs on behalf of its member states.

> [!question]- Q: What is ESA's most significant Earth observation program for open data?
> A: The Copernicus programme, which operates the Sentinel satellite constellation and provides free, open Earth observation data covering land, ocean, atmosphere, and emergency response applications.

> [!question]- Q: Name three Sentinel satellite missions operated under ESA's Copernicus programme.
> A: Sentinel-1 (C-band SAR, radar), Sentinel-2 (multispectral optical, 10m resolution), and Sentinel-3 (ocean and land monitoring, medium resolution).


## Fields of the World

> [!question]- Q: What is Fields of the World (FTW)?
> A: Fields of the World is a global dataset of agricultural field boundaries derived from satellite imagery and various ground-truth sources, intended to support precision agriculture, food security analysis, and agricultural AI model development.

> [!question]- Q: What problem does Fields of the World address?
> A: The lack of a globally consistent, open dataset of individual agricultural field boundaries. Most existing datasets are national or regional; FTW aggregates and harmonizes field boundary data across many countries into a single training and analysis resource.

> [!question]- Q: Who produces Fields of the World?
> A: Fields of the World is a dataset produced by Kerner et al. and associated collaborators, aggregating field boundary data from multiple national and regional sources into a benchmark for agricultural ML.


## Fiona

> [!question]- Q: What is Fiona?
> A: Fiona is a Python GIS library for reading and writing vector geospatial data in formats like GeoPackage and Shapefile. It provides cleaner Python bindings to GDAL/OGR than GDAL's own Python bindings.

> [!question]- Q: What is the relationship between Fiona and GDAL?
> A: Fiona depends on GDAL (specifically the OGR vector component) but provides a more Pythonic API. Geopandas uses Fiona (or pyogrio) under the hood for file I/O.

> [!question]- Q: What is pyogrio relative to Fiona?
> A: pyogrio is a faster drop-in backend for Geopandas that also wraps OGR, offering significantly better performance than Fiona for large datasets.


## FlatBuffers

> [!question]- Q: What is FlatBuffers and who made it?
> A: FlatBuffers is a Google serialization library designed for zero-copy access — data is read directly from the buffer without parsing it into intermediate objects, by computing field offsets and reading bytes directly.

> [!question]- Q: How does FlatBuffers differ from Protocol Buffers (Protobuf)?
> A: Protobuf is designed for network messages and RPC, requires parsing the whole message into native objects before field access, and produces a smaller wire format. FlatBuffers allows direct field access without a parse step, is faster for reads, and is designed for performance-critical systems like game engines.

> [!question]- Q: Why does FlatGeobuf use FlatBuffers?
> A: When doing a spatial query, most features are skipped entirely. FlatBuffers' zero-copy access means you only pay to read the fields you actually access (bbox, geometry bytes), not deserialize the entire feature — a significant win for spatially-filtered reads.


## FlatGeobuf

> [!question]- Q: What is FlatGeobuf?
> A: A binary file format for geographic vector data (points, lines, polygons) designed to be fast, simple, and cloud-friendly. It is essentially a binary encoding of a GeoJSON-like feature collection, built on FlatBuffers, with an optional Hilbert R-Tree spatial index.

> [!question]- Q: What is FlatGeobuf's key cloud-native feature?
> A: HTTP Range Requests combined with the packed spatial index: a client first reads the header and spatial index, queries the index for a bounding box to get byte offsets of matching features, then fetches only those bytes. For a 10GB global file, a small-area query might transfer only a few KB.

> [!question]- Q: How does FlatGeobuf compare to GeoJSON and Shapefile?
> A: FlatGeobuf is 5–10x smaller and faster to read than GeoJSON (but not human-readable). Unlike Shapefile, it is a single file with no 2GB size limit, no 10-character field name limit, and no column type restrictions.

> [!question]- Q: What are FlatGeobuf's main weaknesses?
> A: No attribute (non-spatial) indexing — filtering by non-spatial attributes requires a full scan. It is append-only (no in-place edits; the whole file must be rewritten). Less tooling than Shapefile or GeoPackage.

> [!question]- Q: How does FlatGeobuf compare to GeoParquet?
> A: Both are for vector data. FlatGeobuf is row-oriented and best for reading whole features (spatial queries, streaming). GeoParquet is columnar and best for analytics (aggregations, column-level queries).


## Footprint

> [!question]- Q: What is a satellite image footprint?
> A: The geographic polygon (area) covered by a single scene or image acquisition — the swath width times the along-track length captured in one satellite pass. It is a discrete bounded polygon.

> [!question]- Q: How does a footprint differ from a swath?
> A: The swath is the ongoing continuous strip imaged as the satellite moves along its ground track (it has width but indefinite length). A footprint is one discrete chunk of the swath — the area captured in a single acquisition.


## Gaussian Splatting

> [!question]- Q: What is Gaussian Splatting (3DGS) and what problem does it solve?
> A: 3DGS is a technique for novel view synthesis — given photographs from known camera positions, it reconstructs a scene so it can be rendered from any new viewpoint in real time. It was introduced in 2023 as an alternative to Neural Radiance Fields (NeRFs).

> [!question]- Q: How does 3DGS represent a scene?
> A: As a collection of millions of 3D Gaussian ellipsoids (semi-transparent colored blobs), each with a position (XYZ), covariance (shape and orientation), color, and opacity. Rendering projects these Gaussians onto the image plane, sorts by depth, and alpha-composites them.

> [!question]- Q: What are the key advantages of 3DGS over NeRFs?
> A: Real-time rendering (NeRFs require a neural network forward pass per ray; 3DGS projects and composites Gaussians natively on the GPU), explicit geometry (inspectable and manipulable unlike a black-box NN), comparable or better visual quality, and faster training (minutes to hours vs. days).

> [!question]- Q: What is 3DGS's current limitation for geospatial applications?
> A: 3DGS works well for bounded scenes (a building, a neighborhood), but does not yet scale to city or regional extent the way LiDAR point clouds or photogrammetric DEMs do. Scaling is an active area of research.


## GEO-Bench

> [!question]- Q: What is GEO-Bench and who created it?
> A: GEO-Bench is a standardized benchmark for evaluating geospatial foundation models (GFMs), released in June 2023 by researchers from ServiceNow, TU Munich, Stanford, MIT, ETH Zurich, and ASU. It fills a gap where existing CV benchmarks (ImageNet, COCO) do not reflect real Earth observation challenges.

> [!question]- Q: What tasks does GEO-Bench include?
> A: Six classification tasks and six segmentation tasks, carefully curated to be relevant to Earth monitoring and well-suited for model evaluation.

> [!question]- Q: What are the main limitations of GEO-Bench?
> A: Image-level labels on some tasks (not always pixel-precise), no temporal modeling tasks (all datasets are single-timestamp, so models with strong temporal pretraining can't demonstrate that advantage), and no multi-modal tasks (single-sensor only, so multi-source models don't get credit for radar/LiDAR fusion).


## GeoChat

> [!question]- Q: What is GeoChat?
> A: GeoChat is a multimodal large language model (MLLM) adapted for remote sensing and geospatial imagery, enabling visual question answering, scene description, and spatial reasoning over satellite and aerial images.

> [!question]- Q: What problem does GeoChat address?
> A: General-purpose vision-language models (like LLaVA) perform poorly on remote sensing imagery because they are trained on natural photos. GeoChat is fine-tuned specifically on remote sensing data to answer questions like "how many buildings are in this image" or "describe the land use in this scene."


## GeoCLIP

> [!question]- Q: What is GeoCLIP?
> A: GeoCLIP is a CLIP-based model adapted for geolocation — given a ground-level or aerial photograph, it predicts where on Earth the image was taken by learning embeddings that align image features with geographic coordinates.

> [!question]- Q: How does GeoCLIP differ from standard CLIP?
> A: Standard CLIP aligns image and text embeddings. GeoCLIP aligns image embeddings with geographic location embeddings, enabling geo-localization from visual content rather than text matching.


## Geocode

> [!question]- Q: What is geocoding?
> A: Geocoding is the process of converting a human-readable address or place name into geographic coordinates (latitude, longitude). Example: "1600 Pennsylvania Ave NW, Washington DC" → (38.8977, -77.0365).

> [!question]- Q: What is reverse geocoding?
> A: The opposite of geocoding — converting geographic coordinates into a human-readable address or place description. Example: (34.052, -118.243) → "Downtown Los Angeles, CA".

> [!question]- Q: What are common use cases for forward and reverse geocoding?
> A: Forward geocoding: converting address records with missing coordinates, or powering a map search bar where users type addresses. Reverse geocoding: labeling GPS tracks with street addresses, or determining what place a user is currently near.


## Geocoding Service

> [!question]- Q: What are the main steps in how a geocoding service works internally?
> A: Parsing (decompose input into components), normalization (standardize abbreviations, fix misspellings), candidate retrieval (search reference address database), ranking and disambiguation (score candidates by match quality), and interpolation (estimate position along a street segment for addresses not in the database).

> [!question]- Q: Name three open-source geocoding services.
> A: Nominatim (built on OpenStreetMap data, the go-to for open-source projects), Pelias (OSM-based, designed for self-hosting, more performant than Nominatim at scale), and Photon (lightweight, built on OSM with Elasticsearch, fast and simple to deploy).

> [!question]- Q: What is What3Words and why is it controversial?
> A: What3Words divides the world into 3×3m squares, each assigned a unique 3-word combination (e.g., "filled.count.soap"), designed for communicating precise locations without formal addresses. It is controversial because it is proprietary and not open, creating lock-in for emergency services and others who adopt it.


## Geographic Information Systems

> [!question]- Q: What is GIS (Geographic Information Systems)?
> A: GIS is the framework of hardware, software, and data for capturing, storing, analyzing, managing, and presenting geographic and spatially referenced data. It enables spatial analysis, map creation, and location-based decision making.

> [!question]- Q: What are the five common pitfalls in GIS work?
> A: Swapped coordinates (lat/lon reversed), wrong EPSG assumptions (treating coordinates in one CRS as if they were another), antimeridian issues (polygons crossing the 180° meridian), precision problems (storing unnecessary decimal places), and NULL geometry records (often appearing at (0,0) in the Gulf of Guinea).

> [!question]- Q: What are the key Python libraries for geospatial work?
> A: Geopandas (Pandas + geometry column), Shapely (geometry primitives), pyproj (CRS transformations), rasterio (raster I/O), Fiona (vector file I/O), h3 (H3 indexing), rio-cogeo (COG creation), and SQLAlchemy + GeoAlchemy2 (PostGIS ORM).


## GeoJSON

> [!question]- Q: What is GeoJSON?
> A: GeoJSON is JSON with a geographic extension — a standard format for encoding geographic features (points, lines, polygons) and their properties as JSON. It is the lingua franca of web GIS.

> [!question]- Q: When should you use GeoJSON vs. vector tiles?
> A: Use GeoJSON for small-to-medium datasets (under ~10MB) and bounded queries (e.g., the 50 nearest incidents to a click). Use vector tiles for large datasets rendered on a map — tiles stream only the data visible in the current viewport, preventing the browser from having to parse millions of features at once.

> [!question]- Q: What is the structure of a GeoJSON object?
> A: A FeatureCollection containing Features, where each Feature has a geometry (with type and coordinates) and a properties object for attributes.


## GeoPackage

> [!question]- Q: What is GeoPackage and what format is it based on?
> A: GeoPackage (.gpkg) is a modern vector data format based on SQLite, developed by the OGC as an open replacement for Shapefile. It stores vector features, raster tiles, and raw data in a single portable SQLite database file.

> [!question]- Q: What are the main advantages of GeoPackage over Shapefile?
> A: GeoPackage is a single file (vs. the .shp/.dbf/.prj/.shx bundle), supports multiple layers, handles projections properly, has no 2GB file size limit, and has no 10-character field name restriction.

> [!question]- Q: Why is GeoPackage not cloud-optimized?
> A: Because it is stored as a SQLite database, the entire file must be downloaded to read any part of it, and it requires local file access rather than supporting HTTP range requests from object storage.


## GeoParquet

> [!question]- Q: What is GeoParquet?
> A: A cloud-optimized file format for storing geospatial vector data (points, lines, polygons) in Apache Parquet. It adds conventions for encoding geometries (as WKB) in a geometry column, along with metadata like CRS and bounding box.

> [!question]- Q: What are the key advantages of GeoParquet?
> A: It inherits Parquet's columnar analytics performance for geospatial data. DuckDB with its spatial extension can query a 10GB GeoParquet file with spatial filters in seconds without loading it fully into memory. It is widely used for large open datasets (Overture Maps, Microsoft Building Footprints, Google Open Buildings).

> [!question]- Q: What is a key drawback of GeoParquet for map rendering?
> A: No concept of overviews — to render a zoomed-out view of the whole dataset, you may need to download 100MB+ to the browser. For map rendering at scale, vector tiles are more appropriate; GeoParquet is analytics-first.

> [!question]- Q: How is a Parquet file internally structured?
> A: A Parquet file contains row groups (logical groups of rows), each consisting of column chunks (contiguous column data). Metadata at the end of the file records byte ranges and min/max statistics for each column chunk, enabling both range requests and predicate pushdown.

> [!question]- Q: Does GeoParquet support spatial indexes?
> A: Not natively within the Parquet format itself. Parquet does not have built-in spatial indexing, though data can be spatially partitioned (sorted by Hilbert curve or H3 cell) to improve spatial query performance through data skipping.


## Georef

> [!question]- Q: What is a geographic reference system (Georef)?
> A: Georef (World Geographic Reference System) is a grid-based geographic reference system developed by the US military for air navigation and targeting, encoding any point on Earth as an alphanumeric string based on a 15-degree quadrangle grid, subdivided into progressively finer cells.

> [!question]- Q: How does Georef differ from standard latitude/longitude?
> A: Instead of numeric degrees, Georef uses letters to identify quadrangles at different levels of precision, producing short human-readable codes (e.g., MKQG) that are easier to communicate verbally than decimal coordinates.


## Georeference

> [!question]- Q: What does it mean to georeference a dataset?
> A: Georeferencing aligns a digital image or raster dataset (such as a scanned map or aerial photo) to a known coordinate system, allowing it to be accurately positioned on the Earth's surface.

> [!question]- Q: What are Ground Control Points (GCPs) and how are they used in georeferencing?
> A: GCPs are physical locations with precisely known real-world coordinates. They are identified in both the image and the ground, then used as correspondences to compute a transformation that warps the image to match real-world coordinates.

> [!question]- Q: What is Direct Georeferencing?
> A: Assigning real-world coordinates to sensor data using only the sensor's own position (GPS) and orientation (IMU) measurements, with no Ground Control Points required.


## GEOS

> [!question]- Q: What is GEOS?
> A: GEOS (Geometry Engine – Open Source) is a C library that implements computational geometry operations on 2D vector features — intersection, union, distance, containment, etc. It implements the JTS (Java Topology Suite) specification in C.

> [!question]- Q: What major geospatial tools use GEOS under the hood?
> A: PostGIS (all spatial relationship functions like ST_Intersects, ST_Contains are GEOS), Shapely (which is essentially a Python wrapper around GEOS), and QGIS.

> [!question]- Q: What is the relationship between GEOS, GDAL, and PROJ?
> A: They are the three C libraries that underpin almost all geospatial software. GDAL handles reading/writing raster and vector data formats, GEOS handles computational geometry operations on 2D features, and PROJ handles coordinate reference system transformations.


## Geospatial Data Abstraction Library

> [!question]- Q: What is GDAL and what does it stand for?
> A: GDAL stands for Geospatial Data Abstraction Library. It is a foundational C library for reading and writing both raster and vector geospatial data formats, providing a unified API across hundreds of formats.

> [!question]- Q: What are the two components of GDAL?
> A: GDAL proper (handles raster formats: GeoTIFF, COG, HDF5, NetCDF, Zarr, etc.) and OGR (handles vector formats: Shapefile, GeoJSON, GeoPackage, PostGIS, etc.).

> [!question]- Q: What are the key GDAL command-line tools?
> A: gdalinfo (inspect raster), gdal_translate (convert raster formats), gdalwarp (reproject/warp rasters), ogr2ogr (convert vector formats), and ogrinfo (inspect vector files).

> [!question]- Q: What Python libraries wrap GDAL?
> A: rasterio (wraps GDAL raster), Fiona and pyogrio (wrap OGR for vector). QGIS, PostGIS, and MapServer also use GDAL under the hood.


## Geospatial Embedding

> [!question]- Q: What is a Geospatial Embedding (also called a Geospatial Foundation Model or GFM)?
> A: A large model pretrained on massive satellite imagery datasets to learn general representations of Earth's surface, which can then be fine-tuned or used as embeddings for downstream Earth observation tasks without training from scratch.

> [!question]- Q: What is the core idea behind geospatial foundation models?
> A: Instead of training a new model for every EO task, pretrain once on large-scale satellite imagery to learn compressed, semantically meaningful representations (embeddings), then fine-tune for specific downstream tasks.

> [!question]- Q: What is the key research question for geospatial foundation models?
> A: Which pretext task(s) during pretraining provide useful signal for the widest variety of downstream EO tasks (classification, segmentation, change detection, etc.)?

> [!question]- Q: Name some examples of geospatial foundation models / embeddings.
> A: AlphaEarth Foundations (AEF), OlmoEarth, Clay, TESSERA, and Prithvi (NASA/IBM).


## Geospatial Index

> [!question]- Q: Why can't standard B-Tree indexes handle spatial queries efficiently?
> A: B-Trees work on one-dimensional ordered values. Spatial data is inherently two-dimensional — you cannot sort points by latitude and longitude simultaneously in a way that preserves spatial proximity.

> [!question]- Q: What is an R-Tree and why is it the dominant spatial indexing structure?
> A: An R-Tree groups nearby geometries into minimum bounding rectangles (MBRs), then recursively groups those MBRs into larger MBRs. Queries traverse the tree, pruning entire subtrees whose MBR doesn't intersect the query rectangle. It handles all geometry types and all query types (intersection, containment, distance).

> [!question]- Q: How does PostGIS implement spatial indexing?
> A: PostGIS uses R-Trees implemented on top of the GiST (Generalized Search Tree) framework in PostgreSQL. GiST is a PostgreSQL extension framework for building custom index types; PostGIS plugs spatial logic into it.

> [!question]- Q: What is two-phase spatial query execution?
> A: Phase 1 (index): Use the spatial index to find candidate geometries whose MBR intersects the query — fast but imprecise. Phase 2 (filter): Test the actual geometry against the query predicate exactly. The index provides candidates cheaply; the exact test eliminates false positives.


## Geostationary Orbit

> [!question]- Q: What is a geostationary orbit?
> A: A circular orbit at exactly 35,786 km altitude directly above the equator, where the orbital period is 24 hours — matching Earth's rotation. From the ground, a geostationary satellite appears stationary.

> [!question]- Q: What are the main tradeoffs of geostationary orbit?
> A: Advantages: Persistent staring at a fixed region, no need to track the satellite. Disadvantages: Very high altitude means lower image resolution, higher communication latency, and no coverage near the poles (above ~70° latitude).

> [!question]- Q: What is the relationship between geostationary and geosynchronous orbits?
> A: Geostationary is a subclass of geosynchronous. Both are at 35,786 km altitude. A geosynchronous orbit matches Earth's rotation period (24h) but may be inclined to the equator (causing a figure-8 ground track). A geostationary orbit is geosynchronous AND directly over the equator, maintaining a fixed point over Earth.

> [!question]- Q: How many geostationary satellites are needed for global coverage?
> A: Three satellites spaced 120° apart can cover the entire Earth (each covering more than one-third of the surface).


## Geosynchronous Orbit

> [!question]- Q: What is a geosynchronous orbit (GSO)?
> A: An orbit at 35,786 km altitude where the orbital period matches Earth's rotation (24 hours). Unlike geostationary orbit, a geosynchronous orbit may be inclined to the equator, causing the satellite to trace a figure-8 path (analemma) over the same ground longitude each day.

> [!question]- Q: What are legitimate use cases for an inclined geosynchronous orbit that is not geostationary?
> A: Better high-latitude coverage (geostationary satellites have a poor view above ~60° latitude), antenna footprint shaping (slight inclination shifts coverage northward/southward across the day), and graveyard/transition orbits for end-of-life satellites. In practice, intentional non-geostationary geosynchronous orbits are rare.


## GeoTIFF

> [!question]- Q: What is a GeoTIFF?
> A: A TIFF file with additional tags encoding geospatial metadata: the coordinate reference system (CRS/projection), the geotransform (mapping from pixel to geographic coordinates), and datum/ellipsoid information. It is a regular TIFF that knows where on Earth it lives.

> [!question]- Q: How does a GeoTIFF differ from a plain TIFF?
> A: A plain TIFF viewer shows only the image. A geospatial tool (GDAL, QGIS, rasterio) also reads the embedded spatial metadata to position the image correctly on Earth. The underlying file structure is identical; the geospatial info is stored in TIFF tags.

> [!question]- Q: What is a Cloud-Optimized GeoTIFF (COG)?
> A: A GeoTIFF organized so that internal tiles and overview levels are arranged to support efficient HTTP range requests — allowing clients to fetch only the portion of the image they need without downloading the whole file.


## Geotransform

> [!question]- Q: What is a geotransform?
> A: A set of six numbers that describe where a raster image lies within its coordinate reference system — its resolution and real-world location. Together with a projection definition, it maps each pixel to an accurate real-world coordinate.

> [!question]- Q: What do the six geotransform parameters represent?
> A: Typically: (1) top-left X coordinate, (2) pixel width, (3) row rotation (0 for north-up), (4) top-left Y coordinate, (5) column rotation (0 for north-up), (6) pixel height (negative for north-up images).


## GL Transmission Format

> [!question]- Q: What is glTF (GL Transmission Format) and who developed it?
> A: glTF is an open 3D file format from the Khronos Group designed for efficient runtime delivery of 3D content to the GPU. It is often called "the JPEG of 3D." It consists of a JSON scene description file plus a .bin binary for geometry data, or a single GLB binary variant.

> [!question]- Q: What rendering model does glTF use?
> A: Physically-Based Rendering (PBR), enabling physically realistic material and lighting representation.

> [!question]- Q: What are glTF's main geospatial relevance applications?
> A: 3D Tiles 1.1 uses glTF as its universal tile content format; photogrammetry software exports to glTF; BIM workflows convert IFC to glTF for web visualization; Cesium, Three.js, and deck.gl all load glTF natively.


## Global Ecosystem Dynamics Investigation

> [!question]- Q: What is GEDI and what does it measure?
> A: GEDI (Global Ecosystem Dynamics Investigation) is NASA's spaceborne LiDAR instrument mounted on the International Space Station (ISS), specifically designed to measure global forest structure — canopy height, vertical structure, and biomass estimates.

> [!question]- Q: Why is GEDI mounted on the ISS rather than a dedicated satellite?
> A: Mounting on the ISS allowed GEDI to be launched and operated at much lower cost than a dedicated mission. The ISS orbit (~51.6° inclination) also covers a large portion of the world's tropical and temperate forests.

> [!question]- Q: What is GEDI's primary scientific use case?
> A: Quantifying forest carbon stocks and biomass globally, which is critical for carbon cycle modeling and climate research. It provides the first spaceborne measurements of 3D forest structure at global scale.


## Global Entity Reference System

> [!question]- Q: What is GERS (Global Entity Reference System)?
> A: An open framework introduced by the Overture Maps Foundation for assigning stable, globally unique identifiers to real-world geographic features (buildings, roads, places, boundaries), enabling consistent cross-dataset referencing without fuzzy matching or geometry inspection.

> [!question]- Q: What problem does GERS solve?
> A: The same physical feature (e.g., the Empire State Building) exists in many datasets (OSM, Google Places, Foursquare, NYC building database) each with their own internal IDs that mean nothing to other datasets. GERS provides a common identity layer so datasets can say "this is the same thing."

> [!question]- Q: What are the properties of GERS IDs?
> A: They are globally unique, stable across dataset releases, persistent through geometry/attribute edits, and 128-bit identifiers encoded as opaque strings (meaning should not be parsed from the string itself).


## Global Navigation Satellite System

> [!question]- Q: What is GNSS?
> A: GNSS (Global Navigation Satellite System) is the generic term for any satellite-based positioning, navigation, and timing system. Multiple nations operate independent GNSS constellations.

> [!question]- Q: Name four major GNSS constellations and their operators.
> A: GPS (United States), GLONASS (Russia), Galileo (European Union), and BeiDou (China).

> [!question]- Q: What is the difference between GPS and GNSS?
> A: GPS is one specific GNSS — the US-operated system. GNSS is the umbrella term for all satellite navigation systems. Modern receivers typically use signals from multiple constellations simultaneously for better accuracy and availability.


## Global Positioning System

> [!question]- Q: What is GPS and who operates it?
> A: GPS (Global Positioning System) is the US-operated GNSS constellation providing positioning, navigation, and timing (PNT) anywhere on Earth, using 24+ satellites in Medium Earth Orbit (~22,200 km).

> [!question]- Q: How does GPS determine position?
> A: Each satellite broadcasts its position and a precise atomic clock timestamp. A GPS receiver measures the time-of-flight of signals from multiple satellites to determine distances (spheres). The intersection of four or more distance spheres resolves 3D position and corrects for receiver clock error — this is trilateration (measuring distances), not triangulation (measuring angles).

> [!question]- Q: What are the accuracy levels of GPS?
> A: Consumer GPS: ~3–5m. WAAS/SBAS (differential correction via ground stations): ~1m. Real-Time Kinematic GPS (RTK): centimeter-level. Precise Point Positioning (PPP): also centimeter-level but slower to converge.


## Globalnaya Navigazionnaya Sputnikovaya Sistema

> [!question]- Q: What does GLONASS stand for and who operates it?
> A: GLONASS stands for Globalnaya Navigazionnaya Sputnikovaya Sistema (Global Navigation Satellite System in Russian). It is the Russian GNSS constellation, operated by the Russian Aerospace Defense Forces, providing positioning and timing globally.

> [!question]- Q: How does GLONASS differ technically from GPS?
> A: GPS satellites use CDMA (Code Division Multiple Access) — all satellites transmit on the same frequencies but use different codes. GLONASS uses FDMA (Frequency Division Multiple Access) — each satellite transmits on a slightly different frequency. GLONASS also uses a slightly different coordinate system (PZ-90 vs. GPS's WGS-84).


## GOES-R

> [!question]- Q: What is GOES-R and who operates it?
> A: GOES-R is the current generation of NOAA's geostationary weather satellites (GOES-16, -17, -18, -19). GOES-R represents a major upgrade over the previous GOES-NOP series in imaging capability and cadence.

> [!question]- Q: What is the ABI (Advanced Baseline Imager) on GOES-R satellites?
> A: The primary imaging instrument on GOES-R satellites. It provides 16 spectral bands covering visible through thermal infrared at 500m–2km resolution depending on band, imaging the full Western Hemisphere disk every 10 minutes, CONUS every 5 minutes, and targeted mesoscale regions every 30–60 seconds.

> [!question]- Q: How is GOES-R data accessed?
> A: Freely and publicly on AWS S3 in NetCDF4 format, with SNS notifications on every new file drop — making it one of the best real-time satellite data pipelines available without authentication or cost.

> [!question]- Q: Which GOES-R satellites are currently operational and where are they positioned?
> A: GOES-16 (GOES-East, 75.2°W — covers eastern US, Atlantic, South America) and GOES-18 (GOES-West, 137.2°W — covers western US, Pacific, Alaska). GOES-17 had a cooling defect that degraded its primary imager.


## Google BigLake

> [!question]- Q: What is Google BigLake?
> A: Google BigLake is Google Cloud's service that extends BigQuery's analytics capabilities to data stored in Google Cloud Storage (GCS) in open formats like Apache Parquet and Apache Iceberg, blurring the line between a data lake and a data warehouse (a lakehouse approach).

> [!question]- Q: What problem does BigLake solve?
> A: Organizations had data in both BigQuery (structured, fast analytics) and GCS (cheap object storage for unstructured/semi-structured data). BigLake allows BigQuery to query GCS data as if it were native BigQuery tables, with fine-grained access control and consistent governance across both.


## Google BigQuery

> [!question]- Q: What is Google BigQuery?
> A: BigQuery is Google Cloud's fully managed, serverless, cloud-native OLAP (analytical) data warehouse. It uses a columnar storage format, separates compute from storage, and scales automatically to query petabytes of data using SQL.

> [!question]- Q: What is BigQuery's pricing model and why does it matter architecturally?
> A: BigQuery charges per byte scanned (on-demand) or by reserved capacity slots. This incentivizes columnar storage and partitioning/clustering to reduce bytes scanned — selecting only needed columns and filtering on partition keys dramatically reduces cost.

> [!question]- Q: What are BigQuery's geospatial capabilities?
> A: BigQuery has a native GEOGRAPHY type and spatial functions (ST_Distance, ST_Intersects, ST_Within, etc.) allowing spatial SQL queries at petabyte scale without a separate GIS system.


## Google Earth Engine

> [!question]- Q: What is Google Earth Engine (GEE)?
> A: A cloud computing platform for planetary-scale geospatial analysis. It provides a massive catalog of satellite imagery and geospatial datasets (Landsat, Sentinel, MODIS, etc.) going back decades, and a cloud computing environment (with JS and Python APIs) where analysis runs directly on that data without downloading it.

> [!question]- Q: Who primarily uses Google Earth Engine?
> A: Researchers, scientists, and governments for large-scale environmental monitoring — land cover change detection, deforestation monitoring, crop mapping, water body tracking, etc.

> [!question]- Q: What is the main limitation of Google Earth Engine for commercial users?
> A: GEE has historically been free for research and non-commercial use but requires a commercial license (and cost) for commercial applications, limiting its adoption in commercial geospatial products.


## Google Open Buildings

> [!question]- Q: What is Google Open Buildings?
> A: A global open dataset of building footprints extracted from satellite imagery using machine learning, released by Google Research in 2021, initially focused on Africa and now approaching global coverage with ~1.8 billion buildings.

> [!question]- Q: How does Google Open Buildings compare to Microsoft Building Footprints?
> A: Google Open Buildings has a significantly larger building count (partly because it covers denser informal settlements and rural compounds in the global south better) and has particularly strong coverage in Africa, South/Southeast Asia, and Latin America — regions severely undermapped in OpenStreetMap and other sources.

> [!question]- Q: In what formats is Google Open Buildings available?
> A: Originally released as CSV with WKT geometries, it is also available as GeoParquet via Google Cloud Storage with a STAC catalog.


## Ground Control Point

> [!question]- Q: What is a Ground Control Point (GCP)?
> A: A physical location on Earth's surface with precisely known real-world coordinates (typically surveyed to centimeter accuracy via RTK GPS, dGPS, or total station), used to georeference and orthorectify imagery by providing image-to-ground coordinate correspondences.

> [!question]- Q: What errors do GCPs correct in satellite and drone imagery?
> A: GPS position errors, attitude/orientation errors from the IMU, terrain distortion (assuming flat Earth), lens distortion, and atmospheric refraction — all of which cause pixels to be labeled at incorrect real-world coordinates.

> [!question]- Q: What is the difference between a Control Point and an Independent Check Point (ICP)?
> A: Control Points are used in the processing algorithm to correct/warp the data. ICPs are surveyed to the same accuracy but deliberately withheld from processing; after processing, the discrepancy between processed and surveyed ICP positions gives an independent, unbiased estimate of absolute accuracy.

> [!question]- Q: Where should GCPs ideally be placed in a survey area?
> A: At the edges and corners of the survey area, not clustered in the center. GCPs constrain the transformation across the whole area; clustering them in one region leaves the opposite corners poorly controlled.


## Ground Sample Distance

> [!question]- Q: What is Ground Sample Distance (GSD)?
> A: GSD is the real-world size represented by one pixel in a remotely sensed image — the most precise term for what is loosely called "resolution." For example, WorldView-3's 0.31m GSD means each pixel covers a 31cm × 31cm square on the ground.

> [!question]- Q: How does off-nadir angle affect GSD?
> A: GSD increases (degrades) as the satellite tilts further from straight down (nadir). The further off-nadir the image is acquired, the larger the effective GSD, because the same pixel footprint is spread over a larger ground area.


## Ground Station

> [!question]- Q: What is a ground station?
> A: A terrestrial antenna facility that communicates with satellites — receiving data they have collected (downlinking) and sending commands up to them (uplinking). It is a satellite's only phone call home.

> [!question]- Q: Why are polar ground stations (Alaska, Norway, Antarctica) particularly valuable?
> A: Polar-orbiting satellites (LEO, sun-synchronous) pass over the poles on every orbit. A ground station near the poles gets contact windows on nearly every pass, enabling much more frequent data downlink than a mid-latitude station would.

> [!question]- Q: What are the major commercial ground station networks?
> A: Amazon Ground Station (pay-per-minute, cloud-integrated), Azure Orbital (Microsoft's equivalent), and KSAT (Norwegian, dominant in polar coverage). New cloud-integrated models pipe satellite downlink directly into S3 or equivalent object storage in near-real-time.


## Ground Track

> [!question]- Q: What is a satellite's ground track?
> A: The line traced on Earth's surface directly below the satellite as it orbits — a 1D path with no width, determined purely by orbital mechanics.

> [!question]- Q: How does a ground track relate to a swath and a footprint?
> A: The ground track is the centerline path of the satellite. The swath is the strip of surface imaged on either side of the ground track (it has width). The footprint is a single discrete acquisition polygon within the swath.


## Ground-Penetrating Radar

> [!question]- Q: What is Ground-Penetrating Radar (GPR) and what does it measure?
> A: GPR is a near-surface geophysical method that uses radar pulses to image the subsurface, detecting layer boundaries, buried objects, voids, and changes in material properties by measuring two-way travel time of reflected EM waves.

> [!question]- Q: What is the fundamental physics governing GPR penetration?
> A: EM wave velocity in a material depends on its dielectric permittivity (v = c/√ε). Water has very high permittivity (~80), dramatically slowing radar and causing strong reflections. Wet soil gives strong signal but shallow penetration; dry sand gives deep penetration.

> [!question]- Q: What is the frequency-penetration tradeoff in GPR?
> A: Low frequency = deep penetration but low resolution. High frequency = shallow penetration but high resolution. This is the same fundamental tradeoff as SAR frequency bands (X-band vs. P-band).

> [!question]- Q: Why is spaceborne GPR not feasible on Earth?
> A: The atmosphere (water vapor, ionosphere) absorbs the frequencies needed for useful ground penetration. GPR works from the ground, vehicles, or aircraft. Spaceborne radar sounders work on Mars (MARSIS, SHARAD) only because Mars lacks a significant ionosphere and water vapor.
## gzip

> [!question]- Q: What does gzip provide and what algorithm does it use?
> A: gzip is a lossless compression format for general use, based on the deflate algorithm. It is typically used as standalone external compression.

> [!question]- Q: What file extension do gzip-compressed files use?
> A: gzip files end with the `.gz` extension.

> [!question]- Q: How does gzip differ from formats like Zstandard (zstd) in typical geospatial use?
> A: gzip is a standalone external compression tool applied to whole files, whereas formats like Zstandard can be used as internal, chunk-level compression within formats such as Zarr or Cloud-Optimized GeoTIFF.


## H3

> [!question]- Q: What does H3 stand for and who created it?
> A: H3 stands for Hierarchical Hexagonal Indexing. It is a library created by Uber that divides the Earth's surface into hexagonal cells at multiple resolutions.

> [!question]- Q: How is an H3 cell identified internally?
> A: Each H3 cell has a unique 64-bit integer ID, commonly displayed as a hexadecimal string (e.g., `882a100d2dfffff`). The conventional storage choice is TEXT, since Postgres H3 functions and frontend libraries like deck.gl use text.

> [!question]- Q: Why are hexagons used in H3 rather than squares or rectangles?
> A: Each hexagon has exactly 6 equidistant neighbors, unlike squares (which have 8 neighbors at unequal distances). This makes spatial smoothing and neighbor queries consistent, with no gaps or overlaps.

> [!question]- Q: What is the hierarchical relationship between H3 resolution levels?
> A: Every cell at one resolution contains exactly 7 cells at the next finer resolution (e.g., every resolution-8 cell contains exactly 7 resolution-9 cells). Parent lookup is a cheap bit-mask operation.

> [!question]- Q: What approximate area does an H3 resolution-8 cell cover, and why is it a common analysis sweet spot?
> A: Resolution-8 cells cover roughly 0.7 km², equivalent to a few city blocks — fine enough for neighborhood-level analysis while keeping cell counts manageable.

> [!question]- Q: How does H3 compare to Geohash?
> A: Geohash encodes locations as base-32 strings where truncating gives a coarser cell (simpler prefix queries, no extensions needed), but Geohash cells are rectangles that distort near the poles and have the 4/8 neighbor inconsistency that hexagons avoid. H3 is more uniform globally but requires extensions (e.g., h3-pg) in databases.

> [!question]- Q: What is a known global weakness of H3?
> A: The variation in cell size globally is up to 2x between the largest and smallest cell — a limitation since H3 was designed for ridesharing in cities, not truly global uniform analysis.


## HERE Technologies

> [!question]- Q: What is HERE Technologies and what is its primary market?
> A: HERE Technologies is one of the major commercial map data companies (alongside Google Maps and TomTom), primarily used by automotive and enterprise logistics companies.

> [!question]- Q: What is the ownership history of HERE Technologies?
> A: HERE started as Nokia's mapping division and was sold in 2015 to a consortium of German automakers (BMW, Daimler, Audi) who wanted to control the map data layer for autonomous vehicles. It is now a standalone company.

> [!question]- Q: What key data products does HERE Technologies sell?
> A: HERE sells high-precision road network data (turn restrictions, speed limits, lane geometry), POI data, real-time traffic, HD maps for autonomous driving at centimeter-level precision, and geocoding/routing APIs.


## Hierarchical Data Format 5

> [!question]- Q: What does HDF5 stand for and what is it?
> A: HDF5 stands for Hierarchical Data Format 5. It is a binary file format and library for storing large, complex scientific data, often described as a "filesystem inside a file" with hierarchical groups, datasets, and attributes.

> [!question]- Q: What major scientific data format is built directly on top of HDF5?
> A: NetCDF-4 is literally built on top of HDF5, making HDF5 one of the foundational formats in scientific computing.

> [!question]- Q: What features make HDF5 flexible for scientific use?
> A: HDF5 supports arbitrary nesting of groups (like directories), multiple datasets of mixed data types, metadata attributes, chunking and compression, parallel I/O, and single writer/multiple reader access.

> [!question]- Q: What is a key limitation of HDF5 for cloud/object storage use?
> A: HDF5 was designed for local POSIX filesystems and HPC shared storage. On object storage, navigating its internal B-tree structure requires many small range requests, and the library makes many small reads to traverse metadata — it was not designed for parallel object storage semantics.

> [!question]- Q: What is the recommended cloud-native alternative to HDF5?
> A: Zarr provides the same logical model as HDF5 (hierarchical, chunked, compressed arrays) but was designed from scratch for object storage. The community is slowly migrating, but the existing HDF5/NetCDF archive is enormous.


## Hilbert Curve

> [!question]- Q: What is a Hilbert Curve and why is it useful in geospatial indexing?
> A: A Hilbert Curve is a space-filling curve constructed recursively by subdividing a square into quadrants and connecting them in a U-shape. Its key property is locality preservation: points close together in 2D space tend to have close positions along the 1D curve, which improves spatial data access patterns.

> [!question]- Q: How is a Hilbert Curve constructed?
> A: At each level, a square is subdivided into 4 quadrants connected in a U-shape; each quadrant is then recursively subdivided with rotations/reflections to keep the path continuous. At infinite recursion, it fills every point in the square.

> [!question]- Q: Which geospatial grid systems use Hilbert Curves?
> A: Grid systems like S2 Geometry and H3 use Hilbert Curves (or Hilbert-like orderings) to ensure spatially nearby cells are also nearby in storage order, improving query and I/O performance.


## HLS

> [!question]- Q: What does HLS stand for in the remote sensing context?
> A: HLS stands for Harmonized Landsat and Sentinel-2.

> [!question]- Q: What does the HLS product provide?
> A: HLS is NASA's analysis-ready surface reflectance product that normalizes Landsat and Sentinel-2 imagery into a consistent time series, enabling multi-source temporal analysis.


## HTTP Range Request

> [!question]- Q: What is an HTTP Range Request?
> A: An HTTP Range Request is part of the HTTP specification that allows a client to request a specific byte range from a file rather than downloading the entire file.

> [!question]- Q: Why are HTTP Range Requests critical for cloud-native geospatial formats?
> A: They allow partial reads of geospatial data files stored in cloud object storage, meaning only the relevant portion of a file (e.g., a spatial tile or metadata block) needs to be fetched — the foundation of Cloud-Optimized formats like COG and COPC.


## Ibis

> [!question]- Q: What is Ibis in the Python data ecosystem?
> A: Ibis is a Python dataframe library that provides a pandas-like API which compiles down to SQL and runs against a backend database or query engine (e.g., DuckDB, BigQuery, Snowflake, Postgres).

> [!question]- Q: What problem does Ibis solve?
> A: Ibis lets data scientists write Python expressions against a dataframe-like API while Ibis generates the SQL and the database executes it — keeping workflows in Python without sacrificing the performance of a database engine.

> [!question]- Q: What is a popular Ibis pairing for geospatial analysis?
> A: Ibis + DuckDB is a popular combination for analyzing large geospatial datasets locally (including GeoParquet) without spinning up a full Postgres/PostGIS stack.


## Icechunk

> [!question]- Q: What is Icechunk?
> A: Icechunk is a storage engine for Zarr data that adds transactional, versioned, and cloud-native capabilities — enabling features like git-like history, atomic commits, and time travel for large array datasets stored in object storage.


## ICEYE

> [!question]- Q: What is ICEYE and where is it based?
> A: ICEYE is a commercial SAR imagery provider founded in 2014 in Helsinki, Finland, specializing in X-Band Synthetic Aperture Radar imagery.

> [!question]- Q: Why did ICEYE focus on SAR rather than optical imagery?
> A: Finland has persistent cloud cover, long dark winters, and a strategically significant eastern border, making all-weather, day/night SAR not just commercially interesting but nationally strategic. ICEYE had strong Finnish government support early on.

> [!question]- Q: What is the size and type of the ICEYE satellite constellation?
> A: ICEYE operates 30+ satellites as of 2024, targeting 50+, all X-Band SAR. X-Band provides fine spatial detail and is well-suited for urban areas rather than dense vegetation.

> [!question]- Q: What is ICEYE's headline imaging capability?
> A: ICEYE's "Dwell mode" has the satellite maneuver to track a scene and accumulate returns over extended time, achieving approximately 0.25m resolution — best-in-class alongside Capella's Staring Spotlight mode.

> [!question]- Q: What markets has ICEYE expanded into beyond pure data provision?
> A: ICEYE has expanded into the insurance vertical (flood and disaster monitoring), government/defense contracts, and persistent monitoring subscription services.


## Independent Check Point

> [!question]- Q: What is an Independent Check Point (ICP)?
> A: An Independent Check Point is a surveyed ground location used to assess the accuracy of a geospatial product (such as a DEM or orthoimage) that was not used as input during the product's creation, providing an unbiased accuracy estimate.


## Industry Foundation Classes

> [!question]- Q: What does IFC stand for and who maintains it?
> A: IFC stands for Industry Foundation Classes. It is the open standard file format for BIM (Building Information Modeling) data, maintained by buildingSMART International.

> [!question]- Q: What does an IFC file represent?
> A: IFC defines a schema with hundreds of typed entity classes representing a building hierarchy — from IfcProject down through IfcSite, IfcBuilding, IfcBuildingStorey, to individual elements like IfcWall, IfcDoor, IfcWindow, IfcBeam, and IfcColumn.

> [!question]- Q: What information does each IFC element carry?
> A: Each IFC element includes 3D geometry, property sets (e.g., fire rating, acoustic rating, thermal resistance), relationships to adjacent elements, materials, and classification codes (Uniclass, Omniclass).


## Inertial Measurement Unit

> [!question]- Q: What does IMU stand for and what does it measure?
> A: IMU stands for Inertial Measurement Unit. It measures motion — specifically linear acceleration (via accelerometer) and angular rotation rate (via gyroscope) — without reference to any external signal.

> [!question]- Q: How does an IMU determine position?
> A: By integrating acceleration over time to get velocity, then integrating again to get position. Starting from a known position/attitude, an IMU tracks movement purely through math. The limitation is drift — small sensor errors accumulate over time, which is why IMUs are almost always fused with GPS.

> [!question]- Q: Why is an IMU critical for airborne LiDAR?
> A: LiDAR measures distance to the ground, but you also need to know the exact position and orientation of the sensor at every laser pulse. IMU + GPS (direct georeferencing) provides this; without a high-grade IMU, the resulting point cloud would be a smeared, unusable mess.


## InSAR

> [!question]- Q: What does InSAR stand for?
> A: InSAR stands for Interferometric Synthetic Aperture Radar.

> [!question]- Q: What does InSAR measure and how does it work?
> A: InSAR combines two or more SAR radar images of the same area taken from slightly different positions or times, using the phase difference (interference) of radar waves to measure millimeter-scale surface topography changes or ground deformation.

> [!question]- Q: What are common applications of InSAR?
> A: InSAR is used for earthquake deformation mapping, volcano inflation/deflation monitoring, subsidence detection (cities sinking, e.g., Venice, Mexico City, Jakarta), glacier flow measurement, landslide precursor detection, and permafrost freeze/thaw cycle monitoring.


## Interferometry

> [!question]- Q: What is interferometry?
> A: Interferometry is a technique that extracts information by comparing the phase difference between two wave signals that traveled slightly different paths. Because wave phase is exquisitely sensitive to path length differences, it can act as a ruler for millimeter-scale measurements.

> [!question]- Q: What is phase unwrapping in the context of interferometry?
> A: Phase measurements are inherently ambiguous, cycling between 0 and 2π. Phase unwrapping is the mathematical process of converting these ambiguous phase cycles into continuous deformation values — the hard computational problem in InSAR processing.

> [!question]- Q: How is interferometry applied in geospatial remote sensing?
> A: Two SAR radar images of the same area taken at different times are compared. Subtracting their phases produces an interferogram showing phase differences that encode how much the ground moved between passes — enabling millimeter-scale surface deformation detection over hundreds of kilometers.


## International Maritime Organization

> [!question]- Q: What does IMO stand for and what is it?
> A: IMO stands for International Maritime Organization. It is the United Nations specialized agency responsible for regulating shipping, including safety, environmental concerns, and maritime security standards globally.

> [!question]- Q: What is the IMO's role in vessel identification?
> A: The IMO assigns unique IMO numbers to ships — permanent identifiers that stay with a vessel throughout its life regardless of name or flag changes, used in AIS data and maritime tracking.


## Iridium Satellite Constellation

> [!question]- Q: What service does the Iridium satellite constellation provide?
> A: Iridium provides L-band voice and data communications coverage globally to satellite phones, satellite messenger devices, and integrated transceivers. It became operational in 1998.

> [!question]- Q: How many satellites does Iridium have and at what orbit?
> A: Iridium consists of 66 active satellites (plus spares) in Low Earth Orbit at 781 km altitude and 86.4 degrees inclination — a near-polar orbit.

> [!question]- Q: How does Iridium achieve true global coverage?
> A: The near-polar orbit and inter-satellite communication via Ka-band links between satellites provides global service availability regardless of the position of ground stations. Satellites are arranged in 6 orbital planes of 11 satellites each.

> [!question]- Q: What was the "Iridium flare" phenomenon?
> A: The reflective antennas of first-generation Iridium satellites incidentally focused sunlight onto a small area of Earth, momentarily making the satellite appear as one of the brightest objects in the night sky — visible even during daylight.


## Isochrone

> [!question]- Q: What is an isochrone?
> A: An isochrone is a boundary (polygon) enclosing all locations reachable from a given point within a specified travel time or distance. The word comes from Greek: iso (equal) + chronos (time).

> [!question]- Q: What type of problem is computing an isochrone?
> A: Computing an isochrone is fundamentally a network analysis problem — it requires traversing a road or transit network to determine reachable areas within a given time or distance budget.


## Kerchunk

> [!question]- Q: What is Kerchunk and what problem does it solve?
> A: Kerchunk is a Python library that creates reference files to enable cloud-optimized access to traditional geospatial file formats (like NetCDF and HDF5) without needing to create and store copies of the data in a new format.

> [!question]- Q: How does Kerchunk work technically?
> A: Kerchunk generates JSON reference files with key-value pairs that map Zarr metadata paths or chunk paths to either raw data values or a list of (file URL, starting byte, byte length). This lets Zarr-compatible tools read non-Zarr data as if it were a Zarr store, using HTTP range requests.

> [!question]- Q: Which file formats does Kerchunk support?
> A: Kerchunk supports NetCDF/HDF5, GRIB2, and TIFF, providing a unified way to access chunked, compressed, n-dimensional data across these conventional formats in a cloud-native manner.


## Land Surface Temperature

> [!question]- Q: What does LST stand for and what does it describe?
> A: LST stands for Land Surface Temperature. It describes how warm or cold surfaces on Earth are and the related energy and water exchange processes between land and the atmosphere.

> [!question]- Q: What factors influence Land Surface Temperature?
> A: LST is influenced by the albedo (reflectance) of a surface. It also influences the rate and timing of plant growth and is a key variable in urban heat island studies and climate monitoring.


## Landsat

> [!question]- Q: What is Landsat and who operates it?
> A: Landsat is the longest-running satellite imagery program for Earth observation, operated jointly by NASA and the United States Geological Survey (USGS). The first satellite launched in 1972; Landsat 9 launched in September 2021.

> [!question]- Q: What are the key specifications of Landsat 7?
> A: Landsat 7 has 8 spectral bands, spatial resolutions ranging from 15–60m depending on the band, and a 16-day temporal resolution.

> [!question]- Q: What are common uses of Landsat imagery?
> A: Landsat images are used for agriculture, cartography, geology, forestry, regional planning, surveillance, and education.


## LASer

> [!question]- Q: What does LAS stand for and what is it used for?
> A: LAS stands for LASer. It is the standard binary file format for storing LiDAR point cloud data, developed and maintained by ASPRS (American Society for Photogrammetry and Remote Sensing).

> [!question]- Q: What data attributes does each point in a LAS file store?
> A: Each LAS point stores X/Y/Z coordinates, intensity (signal strength), return number, number of returns per pulse, classification (ground/vegetation/building/water/noise), optional RGB color, GPS timestamp, and scan angle.

> [!question]- Q: What is the relationship between LAS, LAZ, and COPC?
> A: LAS is the base format; LAZ is LAS with lossless compression applied (the de facto storage standard). COPC (Cloud-Optimized Point Cloud) is LAZ reorganized as a spatial octree with chunk locations in the header, enabling HTTP range requests for cloud-native access.


## LAStools

> [!question]- Q: What is LAStools and who created it?
> A: LAStools is proprietary LiDAR processing software created by Martin Isenburg (also the creator of LAZ compression). It is the industry standard for production LiDAR processing, known for being faster than alternatives on large datasets.

> [!question]- Q: What are some key tools included in LAStools?
> A: Key tools include `lasground` (ground classification), `lasheight` (height above ground), `lasclassify` (building/vegetation classification), `las2dem` (DEM interpolation), `lasnoise` (noise filtering), `lastile` (tiling large datasets), and `laszip` (LAZ compression/decompression).

> [!question]- Q: What is the licensing situation for LAStools?
> A: LAStools licensing is complicated — some tools are free for non-commercial use, others require paid licenses. Open-source pipelines often use PDAL as an alternative.


## LAZ

> [!question]- Q: What is LAZ and how does it relate to LAS?
> A: LAZ is the compressed version of the LAS point cloud format, applying lossless compression. It has become the de facto standard for LiDAR storage and transfer; there is no reason to store uncompressed LAS anymore since modern tools read LAZ natively.

> [!question]- Q: What is the cloud-native successor to LAZ?
> A: COPC (Cloud-Optimized Point Cloud) is LAZ reorganized as a spatial octree, with chunk locations stored in a VLR header. HTTP range requests can fetch just the points covering a specific spatial region without downloading the whole file.


## Leaflet

> [!question]- Q: What is Leaflet?
> A: Leaflet is a lightweight, open-source JavaScript library for rendering interactive web maps (slippy maps with tiles, markers, popups, and vector overlays) in the browser. It is approximately 42KB, with a simple API.

> [!question]- Q: What are Leaflet's strengths and limitations?
> A: Leaflet handles the 80% case extremely well — it is small, simple, and has a huge install base. It deliberately has no built-in data analysis, 3D, or advanced projections; everything beyond basics requires a plugin.

> [!question]- Q: When would you choose Mapbox GL JS or deck.gl over Leaflet?
> A: Mapbox GL JS (or its open-source fork MapLibre GL JS) is preferred for vector tiles or 3D rendering. deck.gl is better for heavy data visualization of large datasets.


## leafmap

> [!question]- Q: What is leafmap?
> A: leafmap is a Python package for interactive mapping and geospatial analysis with minimal coding in a Jupyter notebook environment. It is aimed at exploratory data analysis, teaching, and demos rather than production use.

> [!question]- Q: What can leafmap do with one line of code?
> A: leafmap can visualize a GeoDataFrame on an interactive map, inspect raster datasets and satellite imagery (including COGs), and perform basic geospatial operations — all with minimal code in a notebook.

> [!question]- Q: What is leafmap NOT intended for?
> A: leafmap is not a production mapping library and is not a replacement for something like MapLibre GL JS + deck.gl in a frontend application.


## Light Detection and Ranging

> [!question]- Q: What does LiDAR stand for and what does it produce?
> A: LiDAR stands for Light Detection and Ranging. It is an active remote sensing technology that measures distance by emitting laser pulses and timing their return, producing a point cloud — a collection of millions to billions of 3D points.

> [!question]- Q: What is direct georeferencing in the context of LiDAR?
> A: Direct georeferencing is the use of GPS + IMU to determine the exact position and orientation of the LiDAR sensor at every laser pulse, converting raw range measurements into real-world coordinates. Without a high-grade IMU, the point cloud would be geometrically unusable.

> [!question]- Q: What is the significance of multiple returns in LiDAR?
> A: A single laser pulse can generate 2–5+ discrete returns as it penetrates a tree canopy — reflecting from leaves, branches, and finally the ground. This multi-return capability makes LiDAR transformative for forestry and terrain mapping where passive optical imagery only sees the top of the canopy.

> [!question]- Q: How does point density affect what you can extract from LiDAR?
> A: At 1 pt/m² you can build a decent DEM; at 8 pts/m² you can reliably map buildings; at 25+ pts/m² you can extract individual trees and measure their dimensions.

> [!question]- Q: What are the main LiDAR platforms?
> A: Airborne LiDAR (dominant; aircraft/helicopter at 500–3000m), drone LiDAR (low altitude, very high density, limited area), terrestrial LiDAR (ground-based tripod, very high density, limited extent), mobile LiDAR (vehicle-mounted for road corridors), and satellite LiDAR (e.g., ICESat-2, GEDI — along-track profiles, not area coverage).

> [!question]- Q: What is the difference between discrete return and full waveform LiDAR?
> A: Discrete return systems record distinct peaks in the return signal (the standard approach, producing individually labeled points). Full waveform systems digitize the entire continuous return signal — much more data, but richer information about vegetation structure, used in research and forestry.


## Lonboard

> [!question]- Q: What is Lonboard?
> A: Lonboard is a Python library for visualizing large geospatial datasets interactively, built on GeoArrow and GeoParquet technologies combined with GPU-based map rendering from deck.gl.

> [!question]- Q: What is the origin of the name "Lonboard"?
> A: Lonboard is a new binding to the deck.gl visualization library. A "deck" is the part of a skateboard you ride on — a fast, geospatial skateboard is a lonboard.


## Low Earth Orbit

> [!question]- Q: What is Low Earth Orbit (LEO) and what altitude range does it cover?
> A: LEO stands for Low Earth Orbit, covering altitudes of approximately 160–2,000 km. Satellites in LEO have orbital periods of roughly 90–120 minutes and account for about 88% of all satellites.

> [!question]- Q: What are the key advantages of LEO for Earth observation?
> A: Close proximity to Earth enables high spatial resolution imagery, low signal latency, and cheaper launch costs compared to higher orbits.

> [!question]- Q: What are the tradeoffs of LEO?
> A: Any single LEO satellite has a narrow swath and revisits a given point only once or twice per day. Atmospheric drag at the lower end (160–300 km) causes orbital decay, requiring periodic boosts or leading to reentry.


## Mapbox GL JS

> [!question]- Q: What is Mapbox GL JS?
> A: Mapbox GL JS is a JavaScript library for rendering interactive maps in the browser using WebGL, allowing smooth zooming, rotation, and 3D rendering of both raster and vector tiles — a key distinction from the canvas/SVG-based Leaflet.

> [!question]- Q: What happened to Mapbox GL JS in 2021 and what was the community response?
> A: In 2021, Mapbox changed GL JS v2.0 to a proprietary license, which fractured the community. MapLibre GL JS was forked as the open-source successor and is now actively maintained.


## Mapbox Tiling Service

> [!question]- Q: What is Mapbox Tiling Service (MTS)?
> A: MTS is Mapbox's hosted, distributed tile-processing service for ingesting custom datasets of any scale into maps. It is the commercial, hosted evolution of Mapbox's open-source Tippecanoe tool.

> [!question]- Q: What does Mapbox use MTS for internally?
> A: Mapbox uses MTS to create and update its global, daily-updating basemap product Mapbox Streets, which serves over 650 million monthly active users for customers like Facebook, Snap, the Weather Channel, Tableau, and Shopify.


## Mapbox Vector Tile

> [!question]- Q: What does MVT stand for and what is it?
> A: MVT stands for Mapbox Vector Tile. It is the dominant web vector tile format — the map is divided into a grid of tiles at each zoom level, with each tile containing vector features (geometries, properties) clipped and simplified for that tile's bounding box, encoded as Protocol Buffers (binary).

> [!question]- Q: Who maintains the MVT standard now?
> A: MVT is an open standard now maintained by the Open Geospatial Consortium (OGC), widely used beyond Mapbox.

> [!question]- Q: What is the difference between MVT and MBTiles?
> A: MVT is the format spec for a single tile — protobuf-encoded vector features for one Z/X/Y tile. MBTiles is a SQLite container format that stores many tiles (raster or vector) in a single file. MBTiles often stores gzipped MVT blobs in its tiles table.


## Maxar

> [!question]- Q: What is Maxar and what type of satellites does it operate?
> A: Maxar is a satellite imagery company that operates a constellation of Very High Resolution (VHR) commercial optical satellites, including WorldView and GeoEye series satellites.

> [!question]- Q: What is the highest resolution offered by Maxar's satellites?
> A: WorldView-3 provides 0.31m panchromatic resolution and 1.24m multispectral resolution, making it the sharpest commercial satellite in operation.

> [!question]- Q: What are typical use cases and limitations of Maxar imagery?
> A: Maxar imagery is used for defense/intelligence (Maxar is a major NGA contractor), precision mapping, urban planning, infrastructure monitoring, and post-disaster assessment. Limitations include high cost (~$15–30/km²), low revisit for individual satellites, and cloud interference (optical sensor).

> [!question]- Q: What is the Maxar Open Data Program?
> A: The Maxar Open Data Program releases free imagery after major disasters (earthquakes, floods, hurricanes) to support humanitarian response and damage assessment.


## MBTile

> [!question]- Q: What is an MBTile file?
> A: MBTiles is a specification for storing map tiles in a single SQLite database file, created by Mapbox to package thousands or millions of independent tile files into one portable, manageable container.

> [!question]- Q: What is the structure of an MBTiles database?
> A: An MBTiles file is a SQLite database with a `tiles` table that stores each tile keyed by (zoom_level, tile_column, tile_row). Tiles are stored as raw PNG, JPEG, or gzipped MVT.

> [!question]- Q: Why can't MBTiles be served directly from cloud object storage, unlike PMTiles?
> A: SQLite is a file format requiring random byte-level access managed by the SQLite engine running in a local process. It is not a network protocol, so it cannot be accessed directly from object storage without a tile server intermediary.


## Medallion Architecture

> [!question]- Q: What is the Medallion Architecture?
> A: Medallion Architecture is the most common data organization pattern within a Data Lakehouse, using three layers of increasing data quality: Bronze (raw), Silver (cleaned), and Gold (curated).

> [!question]- Q: What do the Bronze, Silver, and Gold layers contain?
> A: Bronze is raw data exactly as it arrived — append-only, preserving full history. Silver is deduplicated, validated, and standardized data that is trustworthy but still relatively raw. Gold contains business-level aggregations, metrics, and features optimized for BI dashboards, ML feature stores, and reporting.

> [!question]- Q: What tools are typically used to transform data between Medallion layers?
> A: Transformation between layers is typically done with Apache Spark or dbt. Each layer is usually stored as Delta Lake or Apache Iceberg tables in object storage.


## Medium Earth Orbit

> [!question]- Q: What is Medium Earth Orbit (MEO) and what altitude range does it cover?
> A: MEO stands for Medium Earth Orbit, covering altitudes of approximately 2,000–35,786 km. Satellites in MEO have orbital periods of 2–24 hours and account for about 2% of all satellites.

> [!question]- Q: What is MEO almost exclusively used for?
> A: MEO is used almost exclusively for navigation/GNSS systems: GPS (20,200 km), GLONASS (19,100 km), Galileo (23,222 km), and BeiDou.

> [!question]- Q: Why do satellite deployments avoid certain altitudes within MEO?
> A: Deployments avoid the Van Allen Radiation Belts, which peak around 1,000–6,000 km and again at 15,000–20,000 km, as intense radiation damages satellite electronics.


## Mercator

> [!question]- Q: What is the Mercator projection and why is it dominant in web mapping?
> A: Mercator (specifically Web Mercator) is a coordinate reference system that is the de facto standard for tiled data in web mapping applications. It rose to prominence when Google Maps adopted it in 2005.

> [!question]- Q: What is the difference between Mercator and Web Mercator?
> A: Web Mercator uses spherical formulas at all scales, while large-scale Mercator maps use an ellipsoidal form of the projection. The discrepancy is imperceptible at global scale but causes slight deviations in local area maps.

> [!question]- Q: What is a key distortion introduced by the Mercator projection?
> A: Mercator preserves angles (conformal) but severely distorts area at high latitudes — making Greenland appear similar in size to Africa, even though Africa is roughly 14x larger.


## Microsoft Building Footprints

> [!question]- Q: What is the Microsoft Building Footprints dataset?
> A: It is a global open dataset of building outlines extracted from satellite imagery using deep learning — one of the largest open building datasets in existence, released as GeoJSON and GeoParquet and free to use.

> [!question]- Q: What is the US coverage of Microsoft Building Footprints?
> A: The dataset covers approximately 130 million buildings across the entire United States, with ongoing global expansion including Canada (~12M buildings) and continent-scale releases.

> [!question]- Q: What are common use cases for Microsoft Building Footprints?
> A: Use cases include population estimation, disaster response (pre-event baseline for damage assessment), tracking urban growth over time, filling OpenStreetMap gaps in undermapped regions, and as training data for other ML models.

> [!question]- Q: How does Google Open Buildings compare to Microsoft Building Footprints?
> A: Both are large open building datasets from satellite imagery using deep learning. Google Open Buildings has particularly strong coverage in Africa and South/Southeast Asia and provides height estimates, making them complementary datasets.


## Microsoft Planetary Computer

> [!question]- Q: What is Microsoft Planetary Computer?
> A: Microsoft Planetary Computer is a cloud platform that hosts large-scale open geospatial datasets alongside Azure compute, enabling analysis without downloading data. It provides a consistent STAC API over datasets stored as COG and GeoParquet.

> [!question]- Q: What datasets are available on Microsoft Planetary Computer?
> A: It hosts 100+ datasets including Landsat Collection 2, Sentinel-2 L2A, NAIP imagery, Copernicus DEM, MODIS products, Microsoft Building Footprints, USGS 3DEP LiDAR in COPC, ERA5 climate reanalysis, and more.

> [!question]- Q: What problem does Planetary Computer solve?
> A: Open geospatial data is theoretically free but practically hard to use — datasets are TBs to PBs, spread across dozens of agencies with inconsistent APIs. Planetary Computer centralizes everything on Azure with a consistent STAC API and co-located compute.


## Military Grid Reference System

> [!question]- Q: What does MGRS stand for and what is it?
> A: MGRS stands for Military Grid Reference System. It is a geocoordinate standard used by NATO militaries to reference locations on Earth using alphanumeric addresses, built on the UTM projection and UPS system for polar regions.

> [!question]- Q: How does MGRS represent a location hierarchically?
> A: MGRS encodes location as Grid Zone Designator (6° × 8° cell, e.g., `33U`), followed by a 100km Square Identifier (two letters, e.g., `XP`), followed by numerical easting and northing digits. More digits give finer precision: 2 digits = 10km, 10 digits = 1m.

> [!question]- Q: How is MGRS used in satellite imagery distribution?
> A: Sentinel-2 imagery is distributed using MGRS tile names (e.g., `T33UXP`), where each tile corresponds to a 100km × 100km UTM square — a pragmatic choice for distributing fixed-size satellite imagery tiles.

> [!question]- Q: Is MGRS a coordinate system or a DGGS?
> A: Neither. MGRS is an addressing convention — a structured, human-readable naming scheme for expressing UTM coordinates as alphanumeric strings, not a coordinate system or a Discrete Global Grid System.


## Moderate Resolution Imaging Spectroradiometer

> [!question]- Q: What does MODIS stand for and what is it?
> A: MODIS stands for Moderate Resolution Imaging Spectroradiometer. It is a satellite-based sensor used for Earth and climate measurements, with two instruments currently in orbit: one on the Terra satellite (1999) and one on the Aqua satellite (2002).

> [!question]- Q: What is the spatial resolution of MODIS?
> A: The majority of MODIS bands have a spatial resolution of 1 km (each pixel = 1 km × 1 km), with some bands at 250m or 500m resolution.

> [!question]- Q: What supplemented MODIS operations after 2011?
> A: Since 2011, MODIS operations have been supplemented by VIIRS (Visible Infrared Imaging Radiometer Suite) sensors, such as the one aboard Suomi NPP, which have similar designs and orbits with subtle differences.


## Mosaic

> [!question]- Q: What is the difference between mosaicking and compositing in remote sensing?
> A: Mosaicking spatially assembles images taken at different locations (roughly the same time) to create a single continuous image covering a larger area. Compositing combines overlapping images of the same location using an aggregation function (e.g., median) to create a single cleaner, often cloud-free image.

> [!question]- Q: What is coregistration and why is it important?
> A: Coregistration is the process of aligning multiple images of the same area so that the same ground point falls on the same pixel across all images. Without it, any cross-image analysis (change detection, InSAR, time series, compositing) produces incorrect results.

> [!question]- Q: Why does InSAR require extremely precise coregistration?
> A: InSAR measures surface deformation through phase differences, and phase is extremely sensitive — a shift of half a wavelength (~3cm for C-Band) completely changes the measurement. InSAR coregistration must be accurate to ~0.001 pixels, far more precise than optical change detection.


## MosaicML

> [!question]- Q: What was MosaicML and what was its business model?
> A: MosaicML was an AI company founded in 2021 by Jonathan Frankle, Naveen Rao, and others. Its business was offering to pretrain LLMs for customers on their platform (essentially as mercenaries), using customer data.

> [!question]- Q: What happened to MosaicML?
> A: MosaicML was acquired by Databricks in June 2023.


## MPT

> [!question]- Q: What does MPT stand for and who released it?
> A: MPT stands for Mosaic Pretrained Transformer. It was released by MosaicML (acquired by Databricks) on May 5, 2023.

> [!question]- Q: What are the key properties of MPT models?
> A: MPT models are open-source, commercially usable LLMs pretrained on 1 trillion tokens. They leverage ALiBi (Attention with Linear Biases) for extended context and FlashAttention for efficiency. Key sizes were MPT-7B and MPT-30B.

> [!question]- Q: What variants of MPT-7B were released?
> A: MPT-7B was released in base, Instruct (short-form instruction following), Chat (dialogue generation), and StoryWriter-65k+ (fine-tuned for 65k-token context on fiction books) variants.


## Multi-task Observation using Satellite Imagery & Kitchen Sinks

> [!question]- Q: What does MOSAIKS stand for and where was it developed?
> A: MOSAIKS stands for Multi-task Observation using Satellite Imagery & Kitchen Sinks. It was developed at UC Berkeley around 2021 as a deliberate anti-complexity statement against deep learning pipelines.

> [!question]- Q: What is the core insight behind MOSAIKS?
> A: Random convolutional filters applied to natural images produce surprisingly informative features — you don't need to learn features, just use enough of them. This is based on the "Random Kitchen Sinks" theory (2007), where random projections into a rich-enough feature space can be linearly separated for a wide variety of tasks.

> [!question]- Q: What does the MOSAIKS pipeline look like?
> A: Given a satellite image patch (256×256 RGB): apply ~8,000 random convolutional filters (fixed, never trained), apply ReLU activation, average pool each filter response to get one scalar, producing a vector of ~8,000 numbers per location. A linear regressor is then trained on top for any specific task.

> [!question]- Q: What makes MOSAIKS well-suited for multi-task learning?
> A: Features are computed once per location and stored. Any new prediction task (forest cover, income, population density, flood risk) just requires training a new linear model on the same stored feature vectors — no re-running imagery pipelines or GPUs needed for feature extraction.


## Multispectral

> [!question]- Q: What is multispectral imagery?
> A: Multispectral satellite imagery is composed of multiple spectral bands (e.g., Red, Green, Blue, Near-Infrared), capturing different portions of the electromagnetic spectrum to enable analysis beyond what the human eye can see.

> [!question]- Q: What is the tradeoff between multispectral and panchromatic imagery?
> A: Multispectral images have multiple bands capturing color/spectral information but at lower spatial resolution, because narrower spectral bands collect less light. Panchromatic images capture a single broad band with no color but at higher spatial resolution.

> [!question]- Q: What is pansharpening?
> A: Pansharpening is the technique of fusing a low-resolution multispectral image with a high-resolution panchromatic image to produce a result that has both high spatial resolution and spectral (color) information.
## Nadir

> [!question]- Q: What is the nadir point in satellite imagery?
> A: The nadir is the point on Earth's surface directly below a satellite. It is the closest point on the ground to the satellite's position.

> [!question]- Q: What does "off-nadir" imaging mean, and what is the tradeoff?
> A: Off-nadir imaging means a taskable satellite has rotated to point at a target that is not directly below it. This enables more flexible targeting but introduces increased distortion and tilt in the resulting imagery.

> [!question]- Q: What is the "look angle" in satellite remote sensing?
> A: The look angle is the number of degrees by which a satellite is pointing off-nadir. A look angle of 0° means the sensor is pointing straight down (nadir); larger angles mean greater tilt.

> [!question]- Q: What is "slew" in the context of satellite operations?
> A: Slew refers to the process of rotating a satellite (changing its attitude) to point at a new target. The slew rate limits how quickly a satellite can be re-tasked to a new location.

> [!question]- Q: What system controls how fast a satellite can be re-tasked?
> A: An Attitude Determination and Control System (ADCS) determines how quickly a satellite can slew and therefore how fast it can be re-tasked to a new target.


## NASADEM

> [!question]- Q: What does NASADEM stand for, and who released it?
> A: NASADEM is a reprocessed and improved version of the original SRTM (Shuttle Radar Topography Mission) global Digital Elevation Model, released by NASA in 2020.

> [!question]- Q: How does NASADEM differ from the original SRTM data?
> A: NASADEM uses the same raw radar data from the 2000 SRTM Shuttle mission but applies two decades of algorithmic improvements, including better void filling, improved phase unwrapping, better absolute accuracy, and improved water body handling.

> [!question]- Q: Is there any reason to use original SRTM data instead of NASADEM?
> A: No. NASADEM is strictly better than the original SRTM — same 30m resolution and same 2000 acquisition date, but cleaner, with fewer voids and higher accuracy.

> [!question]- Q: How does NASADEM compare to Copernicus DEM GLO-30?
> A: Copernicus DEM GLO-30 is still better than NASADEM in almost every way. It was acquired from 2011–2015 (more recent than NASADEM's 2000 baseline) and has largely superseded NASADEM for most use cases.

> [!question]- Q: What is the spatial resolution of NASADEM?
> A: NASADEM has a spatial resolution of 30 meters, matching the original SRTM data it is derived from.


## National Aeronautics and Space Administration

> [!question]- Q: What does NASA stand for?
> A: NASA stands for National Aeronautics and Space Administration. It is a US science and exploration agency.

> [!question]- Q: Which Earth observation satellites does NASA operate?
> A: NASA operates Landsat (jointly with USGS), MODIS/Terra/Aqua, ICESat-2, GEDI (Global Ecosystem Dynamics Investigation), and GRACE, among others.

> [!question]- Q: What core geospatial datasets has NASA developed?
> A: NASA has developed SRTM (Shuttle Radar Topography Mission), NASADEM, and GPM precipitation datasets.

> [!question]- Q: What role does NASA play beyond operating satellites?
> A: NASA funds basic research in remote sensing, climate, and oceanography, and develops foundational datasets used widely across the geospatial and scientific communities.


## National Agriculture Imagery Program

> [!question]- Q: What does NAIP stand for, and who runs it?
> A: NAIP stands for National Agriculture Imagery Program. It is run by the USDA Farm Service Agency (FSA).

> [!question]- Q: What is the spatial resolution and band configuration of NAIP imagery?
> A: NAIP imagery has approximately 0.6–1m spatial resolution and captures four bands: Red, Green, Blue, and Near Infrared (NIR).

> [!question]- Q: How does NAIP's resolution compare to Sentinel-2?
> A: NAIP imagery is sub-meter (~0.6–1m), making it dramatically higher resolution than Sentinel-2's 10m resolution — it is the highest-resolution free optical data available for the US.

> [!question]- Q: What is the temporal coverage pattern of NAIP?
> A: NAIP covers CONUS only, with an irregular state-by-state cycle of roughly a 2–3 year repeat per state. Flights are conducted during the agricultural growing season (leaf-on conditions).

> [!question]- Q: How is NAIP data collected, and what does it cost?
> A: NAIP contracts aerial survey flights using aircraft (not satellites) over the continental US. The data is free and in the public domain, distributed as GeoTIFF tiles by county.

> [!question]- Q: Why is NAIP's NIR band particularly valuable?
> A: The NIR band enables NDVI calculation at sub-meter scale, which is not possible with most free satellite imagery, making NAIP extremely useful for high-resolution vegetation and crop monitoring.


## National Center for Atmospheric Research

> [!question]- Q: What does NCAR stand for?
> A: NCAR stands for the National Center for Atmospheric Research. It is an NSF-funded research center focused on atmospheric science.

> [!question]- Q: Who funds NCAR?
> A: NCAR is funded by the National Science Foundation (NSF).


## National Geospatial-Intelligence Agency

> [!question]- Q: What does NGA stand for, and what is its role?
> A: NGA stands for the National Geospatial-Intelligence Agency. It is a US defense and intelligence agency that produces classified and unclassified geospatial intelligence (GEOINT).

> [!question]- Q: What is WGS84, and which agency manages it?
> A: WGS84 (World Geodetic System 1984) is the global coordinate reference system used as the basis for GPS. The NGA manages and maintains WGS84.

> [!question]- Q: What is DTED, and who produces it?
> A: DTED (Digital Terrain Elevation Data) is a global elevation dataset produced by the NGA. It provides standardized terrain height data used in defense and navigation applications.

> [!question]- Q: What are some unclassified products the NGA publishes?
> A: The NGA publishes unclassified products including the World Port Index, sailing directions, aeronautical charts, and funds foundational geospatial research and standards.

> [!question]- Q: What are NTM products?
> A: NTM (National Technical Means) refers to classified satellite imagery produced by the NGA, used for intelligence and national security purposes.


## National Oceanic and Atmospheric Administration

> [!question]- Q: What does NOAA stand for, and what is its focus?
> A: NOAA stands for the National Oceanic and Atmospheric Administration. It is a US agency focused on oceans and atmosphere, often described as owning "the sky, the coast, and the sea."

> [!question]- Q: What weather satellites does NOAA operate?
> A: NOAA operates GOES-R (geostationary weather satellites) and JPSS (Joint Polar Satellite System) weather satellites.

> [!question]- Q: What geospatial data products does NOAA provide beyond weather?
> A: NOAA provides coastal LiDAR surveys, bathymetry (ocean depth data and nautical charts), tidal datum information, and climate data archives including historical weather stations and sea surface temperatures.

> [!question]- Q: What role does NOAA play in defining "sea level"?
> A: NOAA manages tidal datum and information, which defines what "sea level" means precisely — a critical reference for elevation measurements and coastal mapping.


## National Reconnaissance Office

> [!question]- Q: What does NRO stand for, and what does it do?
> A: NRO stands for the National Reconnaissance Office. It is a US Department of Defense agency and member of the Intelligence Community that designs, builds, launches, and operates the US government's reconnaissance satellites.


## National Science Foundation

> [!question]- Q: What does NSF stand for, and what is its primary function?
> A: NSF stands for the National Science Foundation. It is an independent US federal agency that supports science and engineering research and education across all 50 states and US territories.

> [!question]- Q: Does NSF collect geospatial data directly?
> A: No. NSF funds the science that produces geospatial data rather than collecting it directly. It funds university research, open-source tool development, and platforms like OpenTopography.

> [!question]- Q: What notable geospatial platform does NSF fund?
> A: NSF funds OpenTopography, a LiDAR and DEM data portal hosted at UCSC that provides free access to high-resolution topographic data and on-demand processing tools.

> [!question]- Q: What other geospatial activities does NSF support?
> A: NSF funds Arctic and Antarctic research, development of open-source geospatial tools, and NCAR (National Center for Atmospheric Research).


## Near Infrared

> [!question]- Q: What does NIR stand for, and what is its wavelength range?
> A: NIR stands for Near Infrared. Its wavelength range is approximately 750–900nm, just beyond the end of visible light.

> [!question]- Q: Can human eyes detect near infrared light?
> A: No. Human eyes cannot see NIR, but silicon camera sensors (like those in satellites and cameras) can detect it.

> [!question]- Q: Why do plants reflect NIR strongly?
> A: Healthy leaf cell structure is highly reflective at NIR wavelengths. This strong reflection is a property of healthy vegetation.

> [!question]- Q: How does water respond to NIR radiation?
> A: Water absorbs NIR almost completely, creating a strong contrast with healthy vegetation that reflects it. This contrast is the basis for NDVI and most vegetation indices.


## Network Common Data Form

> [!question]- Q: What does NetCDF stand for, and what is it used for?
> A: NetCDF stands for Network Common Data Form. It is a binary file format for storing multidimensional array data, widely used in climate science, oceanography, meteorology, and atmospheric research.

> [!question]- Q: What is the core data model of NetCDF?
> A: NetCDF is built around labeled, multidimensional arrays with named dimensions (e.g., time, latitude, longitude, pressure level), variables defined over subsets of those dimensions, and metadata attributes attached to variables or the whole file.

> [!question]- Q: What problems does NetCDF have on cloud object storage?
> A: NetCDF was designed for local disk access and has serious problems on object storage: single files can be hundreds of GB, reading small spatial subsets requires many HTTP range requests, files must be fully written before reading, and parallel writes are not supported.

> [!question]- Q: What format was created to address NetCDF's cloud limitations?
> A: Zarr was created to address NetCDF's cloud limitations. It uses the same conceptual model (labeled multidimensional arrays with chunks and metadata) but stores each chunk as a separate object in S3, enabling efficient parallel reads, partial reads, and parallel writes.

> [!question]- Q: Where is NetCDF still dominant despite cloud limitations?
> A: NetCDF remains dominant in traditional HPC/supercomputer environments, while Zarr is increasingly replacing it for cloud-native workflows.


## Normalized Burn Ratio

> [!question]- Q: What does NBR stand for, and what is it used for?
> A: NBR stands for Normalized Burn Ratio. It is a remote sensing index used to identify burned areas and quantify fire severity.

> [!question]- Q: What is the formula for NBR?
> A: NBR = (NIR - SWIR) / (NIR + SWIR), where NIR is Near Infrared and SWIR is Shortwave Infrared. Values range from -1 to 1.

> [!question]- Q: What spectral contrast does NBR exploit?
> A: NBR exploits the contrast between high NIR reflectance from healthy vegetation and high SWIR reflectance from scorched/burned areas.

> [!question]- Q: How do you interpret NBR values?
> A: High values (approximately 0.1 to 1) indicate healthy vegetation, while lower or negative values indicate recently burned areas, bare soil, or low vegetation.


## Normalized Difference Built-up Index

> [!question]- Q: What does NDBI stand for, and what does it measure?
> A: NDBI stands for Normalized Difference Built-up Index. It is a geospatial index used to map and monitor built-up areas such as buildings, roads, and impervious surfaces.

> [!question]- Q: What is the formula for NDBI?
> A: NDBI = (SWIR - NIR) / (SWIR + NIR), where SWIR is Shortwave Infrared and NIR is Near Infrared. Values range from -1 to 1.

> [!question]- Q: What spectral properties does NDBI exploit?
> A: NDBI leverages the high reflectivity of built-up surfaces (concrete, asphalt) in SWIR wavelengths and their low reflectance in NIR, compared to vegetation which strongly reflects NIR.

> [!question]- Q: How do you interpret NDBI values?
> A: Positive values indicate built-up areas (buildings, concrete, asphalt, roads), while negative values indicate non-built-up areas, typically vegetation such as forests and crops.

> [!question]- Q: How does NDBI relate to NDVI?
> A: NDBI and NDVI are essentially inverses of each other in formula structure. NDVI uses (NIR - Red)/(NIR + Red) to highlight vegetation, while NDBI uses (SWIR - NIR)/(SWIR + NIR) to highlight the opposite — built-up, non-vegetated surfaces.


## Normalized Difference Vegetation Index

> [!question]- Q: What does NDVI stand for, and what does it measure?
> A: NDVI stands for Normalized Difference Vegetation Index. It is a remote sensing metric used to quantify vegetation health and density by measuring the difference between NIR (reflected by plants) and Red (absorbed by plants).

> [!question]- Q: What is the formula for NDVI?
> A: NDVI = (NIR - Red) / (NIR + Red). Values range from -1 to +1.

> [!question]- Q: How do you interpret NDVI values?
> A: Healthy vegetation typically falls in the 0.6–0.9 range; sparse vegetation in 0.2–0.5; barren rock, sand, or stressed vegetation in -0.1 to 0.1; and negative values typically indicate water, clouds, or snow.

> [!question]- Q: What are common agricultural applications of NDVI?
> A: NDVI is used to monitor crop health, detect stresses such as pest damage or water shortage before they are visible to the human eye, optimize irrigation, and estimate crop yield.

> [!question]- Q: What is the NDVI Curve?
> A: The NDVI Curve is the NDVI value tracked over time for a location. Different crops peak at different times (e.g., corn and soy have distinct seasonal NDVI profiles), making it useful for crop type identification.

> [!question]- Q: What are the main limitations of NDVI?
> A: NDVI can saturate in dense canopy conditions (loses sensitivity at very high vegetation density), loses accuracy with certain soil types, and is sensitive to atmospheric conditions that can lead to misinterpretation.


## Normalized Difference Water Index

> [!question]- Q: What does NDWI stand for, and what does it detect?
> A: NDWI stands for Normalized Difference Water Index. It is a remote sensing metric used to detect and monitor water content — either surface water bodies or vegetation moisture.

> [!question]- Q: What are the two formulas for NDWI and when is each used?
> A: For surface water bodies: NDWI = (Green - NIR) / (Green + NIR) (McFeeter). For vegetation moisture: NDWI = (NIR - SWIR) / (NIR + SWIR) (Gao, 1996).

> [!question]- Q: How do you interpret NDWI values for water detection?
> A: Values above 0.5 generally represent water bodies; 0 to 0.2 indicates built-up areas; -0.1 to -0.4 indicates green vegetation; values below 0 indicate bare soil, built-up areas, or no moisture.


## OpenMapTiles

> [!question]- Q: What is OpenMapTiles?
> A: OpenMapTiles is an open-source project that defines a schema and toolchain for converting OpenStreetMap data into vector tiles that can be self-hosted or used as a basemap.

> [!question]- Q: What is the core contribution of the OpenMapTiles project?
> A: The core contribution is the schema — a well-defined set of layers (transport, landcover, water, buildings, place labels, etc.) with consistent attributes. It has become the de-facto standard schema for OSM-based vector tiles.

> [!question]- Q: What tile format does OpenMapTiles generate?
> A: OpenMapTiles generates MBTiles. Providers like MapTiler and Stadia Maps serve tiles compatible with the OpenMapTiles schema.

> [!question]- Q: Why did the Protomaps project emerge as an alternative to OpenMapTiles?
> A: OpenMapTiles moved to a dual license in 2019 (free for open source, paid for commercial use). Protomaps emerged as a fully open alternative using the same concept (OSM → PMTiles → self-hostable) with a fully open license and PMTiles format that requires no server process — just static file hosting on S3 or R2.

> [!question]- Q: What problem did OpenMapTiles solve before it existed?
> A: Before OpenMapTiles, self-hosting a vector tile basemap required significant expertise. OpenMapTiles packaged the entire pipeline — from OSM data to rendered vector tiles — into something reproducible.


## OpenStreetMap

> [!question]- Q: What is OpenStreetMap, and how is it often described?
> A: OpenStreetMap (OSM) is a free, editable, openly-licensed map of the entire world built by volunteers. It is often called "the Wikipedia of maps" and has a highly permissive license.

> [!question]- Q: What is the fundamental data model of OpenStreetMap?
> A: OSM is primarily vector data built on nodes (points), ways (lines/polygons), and relations (grouped features), all tagged with key-value metadata (e.g., "highway=residential").

> [!question]- Q: What is the central tension of OSM data quality?
> A: OSM data quality is highly variable. Dense urban areas in wealthy countries often exceed commercial map quality, but rural areas in developing countries can be sparse or outdated. Schema inconsistency is also a problem since anyone can tag features in different ways.

> [!question]- Q: What are the main ways to access OSM data programmatically?
> A: Planet.osm (full weekly dump ~80GB compressed), Geofabrik (pre-cut regional extracts), Overpass API (tag/location queries without full download), and osmnx (Python library for road networks and building footprints).

> [!question]- Q: What is the Humanitarian OSM Team (HOT) and why is it notable?
> A: HOT is an OSM sub-organization focused on disaster response mapping. After the 2010 Haiti earthquake, HOT volunteers remotely mapped Port-au-Prince from satellite imagery in 48 hours, producing the most detailed map of the city ever made and demonstrating that crowdsourced mapping can outpace government or commercial efforts in a crisis.

> [!question]- Q: Why do relief organizations use OSM instead of Google Maps or Apple Maps?
> A: OSM data is openly licensed (can be downloaded, shared, and redistributed freely), can be used offline in the field, is editable (relief workers can add field hospitals, refugee camps, damaged roads), and covers areas that commercial providers neglect because they lack ad revenue potential.


## OpenTopography

> [!question]- Q: What is OpenTopography, and who funds it?
> A: OpenTopography is an NSF-funded platform hosted at UCSC that provides free, easy access to high-resolution topographic data — primarily LiDAR — along with on-demand processing tools.

> [!question]- Q: What global DEMs are available through OpenTopography?
> A: OpenTopography provides access to SRTM, Copernicus GLO-30, ALOS AW3D30, and NASADEM, all through a single API.

> [!question]- Q: What on-demand processing does OpenTopography offer?
> A: Instead of downloading raw point clouds and processing them yourself, OpenTopography lets you draw a bounding box and request derived products such as DEMs directly from LiDAR data.

> [!question]- Q: What is a key limitation of OpenTopography?
> A: On-demand processing has file size limits for free users, and it is not truly cloud-native — users download files rather than doing range-request access in place.

> [!question]- Q: When would a user move beyond OpenTopography to raw data sources?
> A: For serious LiDAR work, users would move to raw LAZ files from USGS 3DEP or state programs, then process with PDAL or lidR to build their own derived products. OpenTopography dramatically lowers the barrier to getting started but has limits for production workflows.


## Orbit

> [!question]- Q: What is an orbit?
> A: An orbit is the curved path that a satellite follows around Earth due to gravitational force.

> [!question]- Q: What are the main classes of Earth orbits by altitude?
> A: Low Earth Orbit (LEO): 160–2,000km; Medium Earth Orbit (MEO): 2,000–35,500km; High Earth Orbit: above 35,500km. Geostationary orbit is at 35,786km.

> [!question]- Q: What is the difference between geosynchronous and geostationary orbit?
> A: Both are at 35,786km altitude where orbital speed matches Earth's rotation. Geostationary is a subclass of geosynchronous where the satellite orbits directly above the equator (on the same plane), maintaining a fixed position over one spot on Earth. Geosynchronous orbits can be tilted above or below the equatorial plane.

> [!question]- Q: Why is MEO commonly used for GPS satellites?
> A: MEO satellites take about 12 hours to complete an orbit and cross the same two equatorial points every 24 hours. This consistent and highly predictable orbit makes it ideal for telecommunications and GNSS (GPS, Galileo) platforms.

> [!question]- Q: What is a polar orbit, and what is it used for?
> A: A polar orbit is inclined nearly 90 degrees to the equatorial plane, allowing the satellite to pass over both poles on each revolution. As Earth rotates below, polar-orbiting satellites can acquire data for the entire globe rapidly, making them ideal for Earth mapping and reconnaissance.


## Orthoimage

> [!question]- Q: What is an orthoimage (orthophoto)?
> A: An orthoimage is an aerial photograph or satellite image that has been geometrically corrected ("orthorectified") to remove distortion caused by camera tilt and terrain relief.

> [!question]- Q: What makes an orthophoto different from a regular aerial photograph?
> A: Unlike an uncorrected aerial photograph, an orthophoto can be used to measure true distances because it is an accurate, uniform-scale representation of the Earth's surface, adjusted for topographic relief, lens distortion, and camera tilt.

> [!question]- Q: What input data is required to create an orthophoto?
> A: A Digital Elevation Model (DEM) or topographic map is required to create an orthophoto, since corrections depend on knowing the varying distance between the sensor and different points on the ground.

> [!question]- Q: How are orthophotos typically used in GIS?
> A: Orthophotographs are commonly used as "map-accurate" background images in GIS systems, since their geometric accuracy makes them suitable for overlaying with vector data and taking measurements.


## Overture Maps Foundation

> [!question]- Q: What is the Overture Maps Foundation?
> A: Overture Maps Foundation is an industry consortium backed by Amazon, Microsoft, Meta, and TomTom, launched in 2022 to produce a free, open, interoperable alternative to proprietary map data and to improve on OpenStreetMap.

> [!question]- Q: What themes (data layers) does Overture Maps publish?
> A: Overture publishes five themes: Places (POIs), Buildings (global footprints), Transportation (roads, connectors, turn restrictions), Administrative Boundaries (countries, states, localities), and Base (land use, land cover, water features).

> [!question]- Q: In what format is Overture data distributed?
> A: Overture data is released as GeoParquet files on Amazon S3, updated regularly.

> [!question]- Q: What is Overture's main criticism of OpenStreetMap?
> A: OSM's schema is inconsistent and data quality varies wildly by region. Overture aims to be more consistent, reliable, and schema-normalized — particularly for building and places layers where OSM has gaps.

> [!question]- Q: Is Overture Maps truly a new mapping project?
> A: Not entirely. Overture ingests and normalizes OSM data as a major input, alongside Microsoft Building Footprints, Meta's road and places data, and TomTom commercial data. It is more accurately described as a data normalization and reconciliation layer on top of existing sources with a consistent schema and open license.


## Overview

> [!question]- Q: What is an "overview" in the context of raster data?
> A: An overview is a downsampled (aggregated) version of raster data intended for visualization, stored as part of a file. Overviews allow reading "zoomed out" data without needing to read and resample full-resolution data.

> [!question]- Q: What file format specification includes overviews?
> A: Overviews are part of the Cloud-Optimized GeoTIFF (COG) specification.

> [!question]- Q: How do overviews relate to tile pyramids?
> A: Overviews and tile pyramids are closely related — both store pre-downsampled versions of data at multiple zoom levels. Overviews typically refer to this concept within a single file, while tile pyramids may span many tiles or files.


## PALSAR

> [!question]- Q: What does PALSAR stand for, and what type of sensor is it?
> A: PALSAR stands for Phased Array type L-band Synthetic Aperture Radar. It is a Japanese radar instrument (not optical), so it can see through clouds and captures surface and subsurface structure.

> [!question]- Q: What makes PALSAR's L-band radar distinctive compared to other SAR sensors?
> A: L-band (~24cm wavelength) is long compared to other common SAR bands, allowing it to penetrate vegetation canopy and interact with large woody structures (trunks, branches) rather than just leaves. By contrast, Sentinel-1's C-band radar only achieves shallow canopy penetration.

> [!question]- Q: What polarization modes does PALSAR support?
> A: PALSAR supports HH (horizontal transmit, horizontal receive — sensitive to surface/trunk scattering), HV (horizontal transmit, vertical receive — sensitive to volume scattering from canopy), and combined HH/HV which provides a much richer forest structure signal.

> [!question]- Q: What is PALSAR particularly useful for in remote sensing?
> A: PALSAR's L-band penetration of the forest canopy makes it especially useful for forest structure mapping, biomass estimation, and detecting subsurface features that shorter-wavelength radar cannot reach.


## Panchromatic

> [!question]- Q: What does "panchromatic" mean in remote sensing?
> A: Panchromatic (Pan) refers to a sensor band that captures a wide range of wavelengths across the visible spectrum in a single grayscale channel, as opposed to multispectral bands that capture narrow wavelength ranges.

> [!question]- Q: Why do panchromatic bands typically have higher spatial resolution than multispectral bands?
> A: Panchromatic bands capture a wide range of wavelengths, collecting more light, which allows the use of smaller pixels (higher spatial resolution). Multispectral bands capture narrow wavelength ranges, collecting less light per band and requiring larger pixels for adequate signal.

> [!question]- Q: What is a typical example of the resolution difference between panchromatic and multispectral on the same satellite?
> A: On Landsat 8, the panchromatic band has 15m resolution while the multispectral bands have 30m resolution.


## PANGAEA

> [!question]- Q: What is PANGAEA in the context of geospatial AI?
> A: PANGAEA is a standardized evaluation protocol (benchmark) for Geospatial Foundation Models (GFMs), introduced in a December 2024 paper. It covers a diverse set of datasets, tasks, image resolutions, sensor modalities, and temporalities.

> [!question]- Q: What problem does PANGAEA address in GFM evaluation?
> A: Existing GFM evaluations were inconsistent and narrow — often using datasets that were too easy or too narrow, lacking diversity in resolution/sensor/temporality, and geographically biased toward North America and Europe. PANGAEA aims to create a robust, globally applicable benchmark.

> [!question]- Q: What was a key finding from the PANGAEA benchmark?
> A: GFMs (Geospatial Foundation Models) do not consistently outperform supervised baselines such as UNet and vanilla ViT, highlighting limitations of current foundation models under various scenarios.


## Pansharpening

> [!question]- Q: What is pansharpening?
> A: Pansharpening is a technique for merging a high-resolution panchromatic (grayscale) band with lower-resolution multispectral (color) bands to produce a high-resolution color image.

> [!question]- Q: What problem does pansharpening solve in satellite imagery?
> A: Satellites face a resolution tradeoff: panchromatic bands are high resolution (more light, smaller pixels) but grayscale, while multispectral bands have color information but lower resolution (less light per band, larger pixels). Pansharpening combines the spatial detail of the panchromatic band with the color information from multispectral bands.

> [!question]- Q: What are common pansharpening algorithms?
> A: Common algorithms include Brovey Transform (simple ratio-based, fast but can distort colors), IHS (Intensity-Hue-Saturation), PCA (Principal Component Analysis), and Gram-Schmidt (more sophisticated, better color preservation, used by ESRI/ENVI).

> [!question]- Q: Is pansharpening suitable for spectral analysis like NDVI?
> A: No. Pansharpening is more of a visualization tool than an analysis tool. Spectral accuracy suffers in the process, making it unsuitable for precise reflectance-based indices like NDVI.

> [!question]- Q: Where is pansharpening commonly applied in practice?
> A: Pansharpening is very common in commercial satellite imagery pipelines. Most "high-res" imagery from Maxar or Planet has been pansharpened.


## Photogrammetry

> [!question]- Q: What is photogrammetry?
> A: Photogrammetry is the science and technology of obtaining reliable information about physical objects and the environment through recording, measuring, and interpreting photographic images or patterns of electromagnetic radiant imagery. Commonly, it refers to the extraction of 3D measurements from 2D data.

> [!question]- Q: What is stereophotogrammetry?
> A: Stereophotogrammetry involves estimating 3D coordinates of points on an object by making measurements in two or more photographic images taken from different positions, exploiting parallax to derive depth.

> [!question]- Q: What is photomapping?
> A: Photomapping is the process of creating a map by assembling aerial or ground-level photographs that are rectified, mosaicked, and annotated with cartographic enhancements like grid lines and place names. It merges photographic detail with traditional map accuracy.


## Physically-Based Rendering

> [!question]- Q: What does PBR stand for, and what is it?
> A: PBR stands for Physically-Based Rendering. It is a shading model that approximates how light actually interacts with surfaces using physically meaningful parameters.

> [!question]- Q: What is the key advantage of PBR over traditional shading models?
> A: A material defined with PBR looks correct under any lighting condition — sunlight, indoor lighting, overcast sky — because it is described in terms of real physical properties rather than arbitrary artistic tweaks.

> [!question]- Q: What is PBR's relevance to geospatial applications?
> A: For photogrammetry meshes and BIM (Building Information Modeling) materials, PBR ensures that textures render plausibly under varied lighting conditions. When a building model is brought into a platform like Cesium and the sun moves across the sky, PBR materials respond correctly to the changing illumination direction.


## Planet Labs

> [!question]- Q: What is Planet Labs, and what is its founding philosophy?
> A: Planet Labs (now just "Planet") is an SF-based Earth observation company founded in 2010 by ex-NASA scientists. Its philosophy is to flip the traditional satellite model: instead of a few expensive, large satellites, build hundreds of cheap small ones.

> [!question]- Q: What are the main satellite types in Planet's constellation?
> A: Planet operates Doves/PlanetScope (~200+ 3U cubesats for daily global 3–4m coverage), SkySat (~21 larger satellites for 0.5m tasked imaging and video), SuperDove (upgraded 8-band Doves for better vegetation analytics), and Pelican (newest generation, targeting 0.3m resolution).

> [!question]- Q: What is the primary competitive advantage of Planet's constellation?
> A: Planet's daily global revisit at 3–4m resolution is unique. No other operator can provide near-daily cloud-free imagery of the entire Earth's landmass.

> [!question]- Q: How does Planet compare to Maxar and Sentinel-2?
> A: Planet provides coarser resolution (3–4m) but daily revisit; Maxar provides sharper imagery (~0.3–0.5m) but infrequent revisit; Sentinel-2 provides free 10m imagery every 5 days.

> [!question]- Q: What is the SkySat capability, and where did it originate?
> A: SkySat is Planet's 50cm tasking capability, providing high-resolution imagery and short video clips from orbit. It was absorbed from Terra Bella/Google when Planet acquired it.


## PlanetScope

> [!question]- Q: What is PlanetScope?
> A: PlanetScope is the largest Earth-observation satellite constellation by satellite count, operated by Planet Labs. It consists of ~200+ small "Dove" cubesats in Low Earth Orbit/Sun-Synchronous Orbit.

> [!question]- Q: What are PlanetScope's key specs?
> A: PlanetScope provides 3–4m multispectral resolution with near-daily global coverage of the entire Earth's landmass.

> [!question]- Q: What use cases is PlanetScope well-suited for?
> A: PlanetScope is ideal for time-series analysis where revisit frequency matters more than resolution — crop monitoring, deforestation alerts, construction tracking, agricultural NDVI time series, and change detection needing daily granularity rather than Sentinel-2's 5-day revisit.

> [!question]- Q: What is a key limitation of PlanetScope?
> A: Its 3–4m resolution is coarser than Maxar, making it less useful for counting or identifying individual objects. Commercial access is also subscription-based and expensive.


## PMTiles

> [!question]- Q: What is PMTiles?
> A: PMTiles is a single-file cloud-native archive format used for vector tiles (and optionally raster tiles), designed to be served directly to clients over a network from object storage using HTTP range requests, without any server-side tile-serving process.

> [!question]- Q: How does PMTiles differ from MBTiles?
> A: MBTiles stores tiles in a SQLite database and requires a server process to serve tiles to clients. PMTiles is serverless — browser clients can fetch tiles directly using HTTP range requests from static file hosting on S3 or R2.

> [!question]- Q: How are tiles organized internally in a PMTiles file?
> A: Tiles are oriented along a Hilbert Curve, meaning spatially nearby tiles are stored near each other in the file. This optimizes spatial locality for visualization use cases where tiles in the same viewport are requested together.

> [!question]- Q: What tool is commonly used to generate PMTiles from vector data?
> A: Tippecanoe is the most common tool for generating PMTiles from vector data.

> [!question]- Q: What is a key limitation of PMTiles for analytical use cases?
> A: PMTiles uses a tile pyramid with multi-resolution overviews, meaning features are dropped at lower zoom levels. This makes it inappropriate for analysis (e.g., you can't get an accurate building count from overview tiles) — it is a lossy technique optimized for interactive visualization, not data analysis.

> [!question]- Q: How do PMTiles, GeoParquet, and COG complement each other?
> A: They are complementary: COG stores raw raster imagery for efficient partial reads; GeoParquet stores vector features (building footprints, POIs) for analytical queries; PMTiles wraps styled map tiles for interactive web visualization.


## Point Data Abstraction Library

> [!question]- Q: What does PDAL stand for, and what is it?
> A: PDAL stands for Point Data Abstraction Library. It is an open-source C++ library with a Python interface that reads and writes virtually every point cloud format (LAS, LAZ, COPC, EPT, PLY, etc.) and provides a pipeline-based processing framework.

> [!question]- Q: What is PDAL analogous to in the raster world?
> A: PDAL is essentially the GDAL of point clouds — a universal I/O and processing library for point cloud data.

> [!question]- Q: How does PDAL process data?
> A: PDAL uses a JSON pipeline of readers, filters, and writers. A pipeline might read a LAZ file, apply a ground filter, select only ground-classified points, and write a raster DTM at 1m resolution.

> [!question]- Q: What role does PDAL play in the broader LiDAR ecosystem?
> A: PDAL is the foundation that most open-source LiDAR tools are built on or interoperate with.


## Polar Orbit

> [!question]- Q: What is a polar orbit?
> A: A polar orbit is one in which a satellite passes above or nearly above both poles of the Earth on each revolution, with an inclination of about 80–90 degrees to the equatorial plane.

> [!question]- Q: What is a polar orbit commonly used for?
> A: Polar orbits are used for Earth-mapping satellites, reconnaissance satellites, and some weather satellites because they enable global coverage including the polar regions.

> [!question]- Q: What is the relationship between polar orbits and sun-synchronous orbits?
> A: Near-polar orbiting satellites commonly choose a sun-synchronous orbit, where each successive orbital pass occurs at the same local solar time each day. This prevents temporal aliasing where changes due to time-of-day differences are confused with real changes on the ground.

> [!question]- Q: What is a fun fact about launching satellites into polar orbit?
> A: Launching into polar orbit requires a larger launch vehicle than for a near-equatorial orbit at the same altitude because a polar launch cannot take advantage of Earth's rotational velocity as a boost.


## Portable Network Graphics

> [!question]- Q: What does PNG stand for?
> A: PNG stands for Portable Network Graphics. It is a widely used raster image format that supports lossless compression and transparency (alpha channel).


## Post-Processed Kinematic

> [!question]- Q: What does PPK stand for?
> A: PPK stands for Post-Processed Kinematic. It is a GPS/GNSS positioning technique where raw receiver data is post-processed against reference station data after collection to achieve centimeter-level accuracy, as opposed to Real-Time Kinematic (RTK) which computes corrections in real time.


## Potree

> [!question]- Q: What is Potree?
> A: Potree is the dominant open-source WebGL-based point cloud renderer that enables visualization of massive LiDAR datasets (billions of points) interactively in a web browser.

> [!question]- Q: How does Potree handle datasets too large to load into memory?
> A: Potree uses a streaming level-of-detail approach built on the octree structure of Entwine Point Tiles (EPT). When zoomed out, it loads only the coarse root node; as you zoom in, it progressively loads deeper octree nodes for the visible region only.

> [!question]- Q: What is the key principle behind Potree's performance?
> A: At any moment, Potree only loads and renders the octree nodes relevant to the current viewpoint and zoom level; the rest remains on the server. This avoids loading the entire dataset into memory.


## Prefect

> [!question]- Q: What is Prefect?
> A: Prefect is a workflow orchestration tool used to build, schedule, and monitor data pipelines, similar in purpose to Apache Airflow and Dagster.


## Prithvi

> [!question]- Q: What is Prithvi?
> A: Prithvi is a 2023 Geospatial Foundation Model developed by NASA and IBM, with open weights on HuggingFace. It is named after the Sanskrit word for Earth.

> [!question]- Q: What architecture does Prithvi use?
> A: Prithvi uses a Masked Autoencoder (MAE) with a Vision Transformer (ViT) backbone. The pretraining objective is to mask random patches of imagery and reconstruct them.

> [!question]- Q: What data was Prithvi trained on?
> A: Prithvi was trained on HLS (Harmonized Landsat Sentinel-2), NASA's analysis-ready surface reflectance product, with a CONUS-focused training set of approximately 4.2 million image chips.

> [!question]- Q: How does Prithvi's training data scale compare to other geospatial foundation models?
> A: Prithvi's ~4.2M image chips is much smaller than Clay's ~70M or AlphaEarth's ~1B observations.


## Prithvi 2.0

> [!question]- Q: What is Prithvi 2.0, and who developed it?
> A: Prithvi 2.0 is a Geospatial Foundation Model released in December 2024 by IBM and NASA. It is a versatile multi-temporal foundation model for Earth observation applications.

> [!question]- Q: How does Prithvi 2.0's architecture differ from the original Prithvi?
> A: Prithvi 2.0 uses a 3D Masked Autoencoder that extends the standard ViT MAE to spatiotemporal cubes, using 3D patch embedding via Conv3D and treating (time × height × width) as a cube.

> [!question]- Q: What metadata does Prithvi 2.0 incorporate during pretraining?
> A: Prithvi 2.0 embeds metadata including latitude, longitude, year, and day of year as additional tokens. Metadata dropout (p=0.1) during pretraining teaches the model to work with or without location/time information.


## PROJ

> [!question]- Q: What is PROJ, and what does it do?
> A: PROJ is a C library that converts coordinates between Coordinate Reference Systems (CRS). It knows hundreds of map projection functions and can transform a point from one coordinate system to another.

> [!question]- Q: Why is PROJ necessary in geospatial workflows?
> A: Different data sources use different coordinate systems — GPS uses WGS84 (latitude/longitude), city survey data may use a local projection, and satellite imagery may be in UTM. PROJ is needed to put them all in the same coordinate system for analysis.

> [!question]- Q: What are the three foundational C libraries that underpin most geospatial software?
> A: GDAL (raster/vector I/O), GEOS (geometry operations), and PROJ (coordinate transformations). They are the engine under tools like PostGIS, Shapely, and QGIS.

> [!question]- Q: How is PROJ used in PostGIS?
> A: PostGIS's ST_Transform() function calls PROJ under the hood to reproject geometries between coordinate reference systems.


## R-Tree

> [!question]- Q: What is an R-Tree?
> A: An R-Tree is a geospatial index (invented in 1984) that indexes Maximum Bounding Rectangles (MBRs) of geometries. It groups nearby bounding boxes into parent bounding boxes recursively to enable efficient spatial queries.

> [!question]- Q: How does an R-Tree answer a spatial query like "find all points within 500m of City Hall"?
> A: It converts the search radius to a bounding box, then at the root checks which child bounding boxes overlap the search and prunes non-overlapping branches. It recurses only into overlapping branches, doing an exact geometry check only at the leaf level — checking perhaps 200 candidates instead of millions.

> [!question]- Q: How does an R-Tree differ from a QuadTree?
> A: QuadTrees split space into fixed quadrants regardless of data distribution. R-Trees work with flexible, overlapping rectangles that adapt to actual data, making them more efficient for non-uniform distributions and for indexing both points and complex shapes in the same structure.

> [!question]- Q: What is a Hilbert R-Tree?
> A: A Hilbert R-Tree is a variant of the R-Tree that uses Hilbert curve ordering to organize data, improving spatial locality. It is used in FlatGeobuf for file-internal spatial indexes.

> [!question]- Q: What is the main tradeoff of R-Trees?
> A: The flexible overlapping rectangles sometimes force searches through multiple branches of the tree. Modern R-Tree implementations use sophisticated algorithms to balance this overlap against tree depth, optimizing for how databases read data from disk.


## RADARSAT

> [!question]- Q: What is RADARSAT?
> A: RADARSAT is a Remote Sensing Earth observation satellite program overseen by the Canadian Space Agency (CSA). It uses Synthetic Aperture Radar (SAR) to image the Earth's surface.

> [!question]- Q: What are the three generations of RADARSAT?
> A: RADARSAT-1 (1995–2013), RADARSAT-2 (2007–present), and the RADARSAT Constellation Mission (2019–present), which uses three small satellites for improved coverage and resilience.

> [!question]- Q: What improvement did the RADARSAT Constellation Mission bring?
> A: The Constellation uses three small satellites to provide greater coverage while minimizing service interruptions. Its revisit period is 4 days, compared to 24 days for its predecessors.


## Radiant Earth Foundation

> [!question]- Q: What is the Radiant Earth Foundation?
> A: Radiant Earth Foundation is a nonprofit focused on open ML training data for Earth observation. They are deeply embedded in the cloud-native geospatial ecosystem (STAC, COG, open data, humanitarian applications).

> [!question]- Q: What is Radiant MLHub?
> A: Radiant MLHub is Radiant Earth's main product — an open repository of labeled geospatial training datasets for tasks like crop type mapping, flood extent detection, building footprint extraction, and land cover classification. It is often described as "HuggingFace for satellite imagery training datasets, with a STAC API."

> [!question]- Q: Why is Radiant Earth's work important for EO machine learning?
> A: Getting labeled training data is the hardest part of Earth observation ML. Radiant Earth curates, standardizes, and openly hosts datasets so researchers don't have to re-annotate the same imagery repeatedly.

> [!question]- Q: What is Radiant Earth's connection to STAC?
> A: Radiant Earth was a major contributor to the SpatioTemporal Asset Catalog (STAC) specification, and all Radiant MLHub datasets are served via a STAC API.


## RapidEye

> [!question]- Q: What is RapidEye, and who currently owns it?
> A: RapidEye was a constellation of 5 Earth observation satellites deployed in 2008. It was acquired by Planet Labs in 2015 and retired in 2020.

> [!question]- Q: What were RapidEye's key specs?
> A: RapidEye had a 6.5m ground sampling distance (5m orthorectified pixel size), a 77km swath width, a daily off-nadir revisit time, a 630km sun-synchronous orbit, and 5 spectral bands.

> [!question]- Q: What spectral bands did RapidEye capture, and what made it distinctive?
> A: RapidEye captured Blue (440–510nm), Green (520–590nm), Red (630–685nm), Red Edge (690–730nm), and Near IR (760–850nm). The inclusion of a red edge band was distinctive, as it is particularly sensitive to vegetation stress and chlorophyll content.


## Raster

> [!question]- Q: What is raster data in geospatial contexts?
> A: Raster data represents continuous fields as grids of cells (pixels), where each cell holds a numeric value. Examples include satellite imagery (color values per pixel), elevation models (meters above sea level per pixel), and temperature grids.

> [!question]- Q: What is the key difference between raster and vector data?
> A: Vector data is precise at boundaries (a polygon has a definitive edge), while raster data is continuous (values blend smoothly between pixels). Vector data represents discrete features; raster data represents continuous fields.

> [!question]- Q: What is a "band" in raster data?
> A: A band is a separate data layer in a raster file where each band stores one value per pixel. An RGB image has 3 bands, RGBA has 4, and hyperspectral satellite data may have 200+ bands, each capturing a different wavelength of light.

> [!question]- Q: What is a GeoTIFF?
> A: A GeoTIFF is the standard raster format — a regular TIFF image file with embedded geospatial metadata including the Coordinate Reference System and the real-world coordinates of the image corners.

> [!question]- Q: What problem does a Cloud-Optimized GeoTIFF (COG) solve compared to a classic GeoTIFF?
> A: A classic GeoTIFF stores pixels in row order, requiring full decompression to extract a subset. A COG is reorganized with internal tiling (256×256 or 512×512 pixel tiles), overview pyramids at multiple zoom levels, and an index at the start — enabling efficient HTTP range requests to fetch only the specific tile and zoom level needed without reading the full file.

> [!question]- Q: What is Zarr, and when is it preferred over GeoTIFF?
> A: Zarr is a format for chunked, compressed N-dimensional arrays. Unlike COG (which handles 2D grids), Zarr handles arbitrary dimensions (space × time × variable), making it the dominant format for climate science and Earth observation time series. Each chunk is a separate file in object storage, enabling efficient parallel and partial reads in the cloud.
## RasterFlow

> [!question]- Q: What is RasterFlow?
> A: RasterFlow is Wherobots' serverless Earth Observation (EO) inference engine that automates the full pipeline from raw satellite/drone imagery to actionable geospatial insights at planetary scale.

> [!question]- Q: What core problem does RasterFlow solve?
> A: EO analysis traditionally requires rare specialist expertise and expensive custom infrastructure. RasterFlow makes it accessible to general data teams.

> [!question]- Q: What built-in models does RasterFlow include?
> A: It includes Fields of the World (FTW) for crop boundary detection, Meta tree canopy height, road segmentation, and foundation model embeddings (OlmoEarth, Clay).

> [!question]- Q: What does RasterFlow output and where?
> A: It vectorizes model predictions into geometries and writes them as Apache Iceberg tables to S3, integrating with WherobotsDB, Databricks, Snowflake, and BigQuery.

> [!question]- Q: What ML framework does RasterFlow use for distributed inference?
> A: RasterFlow runs PyTorch models (custom or pre-built) distributed across massive geographic areas, enabling country-scale inference in minutes to hours.


## rasterio

> [!question]- Q: What is rasterio?
> A: Rasterio is the primary Python library for reading and writing raster geospatial data, including satellite imagery, aerial photos, and elevation models stored as grids of pixel values.

> [!question]- Q: What underlying engine does rasterio wrap?
> A: Rasterio wraps GDAL (Geospatial Data Abstraction Library), which has existed since the late 1990s and supports roughly 160 raster formats. GDAL's Python bindings are notoriously painful to use, so rasterio provides a clean, Pythonic API.

> [!question]- Q: How is rasterio analogous to Fiona?
> A: Just as rasterio wraps GDAL's raster side with a clean Python API, Fiona wraps GDAL's vector side. They are complementary libraries for the two main types of geospatial data.

> [!question]- Q: What are common operations performed with rasterio?
> A: Opening a GeoTIFF and reading pixel values as NumPy arrays, reprojecting rasters between coordinate systems, clipping a raster to a polygon, reading metadata (CRS, resolution, bounding box, band count), and writing processed arrays back to GeoTIFF.

> [!question]- Q: What is "band count" in the context of rasterio?
> A: Band count refers to how many separate data layers a raster file contains, where each band stores one value per pixel. An RGB image has 3 bands, RGBA has 4, and hyperspectral satellite data might have 200+ bands each capturing a different wavelength.


## Rational Polynomial Coefficients

> [!question]- Q: What does RPC stand for?
> A: RPC stands for Rational Polynomial Coefficients.

> [!question]- Q: What are Rational Polynomial Coefficients used for in remote sensing?
> A: RPCs are a mathematical sensor model used to relate 3D ground coordinates to 2D image pixel coordinates. They are used for orthorectification and photogrammetric processing of satellite imagery without requiring full rigorous sensor models.

> [!question]- Q: What is the advantage of RPCs over rigorous sensor models?
> A: RPCs are a compact, vendor-neutral representation that can be distributed with imagery and used by any software, whereas rigorous sensor models require detailed knowledge of the satellite's internal geometry and are often proprietary.


## Real-Time Kinematic GPS

> [!question]- Q: What does RTK GPS stand for?
> A: RTK GPS stands for Real-Time Kinematic GPS.

> [!question]- Q: What problem does RTK GPS solve compared to standard GPS?
> A: Standard GPS typically achieves meter-level accuracy due to atmospheric errors and satellite geometry. RTK GPS uses a fixed base station with a known position to transmit corrections to a rover receiver in real time, achieving centimeter-level positioning accuracy.

> [!question]- Q: What are common applications of RTK GPS?
> A: RTK GPS is used in precision agriculture (automated machine guidance), land surveying, construction stakeout, drone ground control point collection, and any application requiring centimeter-level positional accuracy.

> [!question]- Q: How does RTK GPS achieve its high accuracy?
> A: RTK uses carrier-phase measurements of the GPS signal rather than just the coarser code-phase measurements. A nearby base station computes the difference between its known position and its measured position, and broadcasts these corrections to the rover.


## Remote Sensing

> [!question]- Q: What is remote sensing?
> A: Remote sensing is the acquisition of information about an object or phenomenon without making physical contact, generally referring to the use of satellite or airborne sensor technologies to detect and classify objects on Earth.

> [!question]- Q: What is the difference between active and passive remote sensing?
> A: Active remote sensing emits a signal (e.g., radar or LiDAR) and detects the reflection, while passive remote sensing detects naturally reflected sunlight or thermally emitted radiation from the Earth. Active sensors can operate day or night and through clouds; passive sensors depend on sunlight and are blocked by clouds.

> [!question]- Q: What are the four types of resolution in remote sensing?
> A: Radiometric resolution (bit depth, number of intensity levels), spatial resolution (pixel size on the ground), spectral resolution (number and width of spectral bands), and temporal resolution (how frequently the same area is revisited).

> [!question]- Q: Why is it difficult to build a sensor with high resolution in all four dimensions?
> A: There are fundamental tradeoffs: achieving high spatial resolution requires a narrower swath, which reduces the area covered per pass and thus reduces temporal resolution. Similarly, narrow spectral bands reduce the energy captured per band, affecting radiometric quality.

> [!question]- Q: What are NASA's data processing levels 0–4?
> A: Level 0 is raw unprocessed data; Level 1a is time-referenced with calibration coefficients appended but not applied; Level 1b has calibration applied; Level 2 is derived geophysical variables at L1 resolution; Level 3 is gridded with spatial/temporal completeness; Level 4 is model output or analysis results derived from lower levels.

> [!question]- Q: What is spectral unmixing?
> A: Spectral unmixing decomposes pixels into fractional abundances of pure spectral signatures called endmembers. Rather than classifying each pixel as one class, it provides a continuous mixture (e.g., a pixel is 60% forest, 30% soil, 10% shadow), which is useful for moderate-resolution imagery where most pixels contain multiple materials.

> [!question]- Q: What is the difference between a Digital Surface Model (DSM) and a Digital Terrain Model (DTM)?
> A: A DSM represents the elevation of the top surface including buildings, trees, and other above-ground objects. A DTM represents bare-earth elevation with buildings and vegetation removed.


## RemoteCLIP

> [!question]- Q: What is RemoteCLIP?
> A: RemoteCLIP is a vision-language foundation model adapted from CLIP for remote sensing, trained to align satellite image embeddings with text descriptions to enable zero-shot and few-shot EO tasks.

> [!question]- Q: What problem does RemoteCLIP address compared to general CLIP?
> A: General CLIP is trained on natural photos and web text, so it performs poorly on satellite imagery with overhead views, unusual object scales, and remote-sensing-specific vocabulary. RemoteCLIP fine-tunes on curated EO image-text pairs to close this domain gap.

> [!question]- Q: What tasks can RemoteCLIP support?
> A: RemoteCLIP supports zero-shot image classification, text-based image retrieval, and scene understanding in remote sensing without task-specific labeled training data.


## Renaissance Philanthropy

> [!question]- Q: What is Renaissance Philanthropy (RenPhil)?
> A: Renaissance Philanthropy is a philanthropic organization focused on funding science and technology initiatives, operating as a donor-advised fund and grantmaker supporting research and innovation projects.

> [!question]- Q: What distinguishes Renaissance Philanthropy's approach?
> A: Renaissance Philanthropy emphasizes high-leverage scientific research and technology development, often funding early-stage work that other funders may overlook, with a focus on outsized societal impact.


## Resolution

> [!question]- Q: What are the four types of resolution in remote sensing?
> A: Radiometric resolution (number of intensity levels a sensor can distinguish, measured in bits), spatial resolution (the real-world size of one pixel), spectral resolution (the number and narrowness of spectral bands), and temporal resolution (how frequently the same location is revisited).

> [!question]- Q: What is the tradeoff between spatial resolution and temporal resolution?
> A: To achieve higher spatial resolution, a satellite must use a narrower field of view and therefore a narrower swath, meaning it covers less area per pass and takes longer to revisit the same location, resulting in lower temporal resolution.

> [!question]- Q: How are multispectral and hyperspectral sensors defined in terms of spectral resolution?
> A: Multispectral sensors have 3–10 bands, while hyperspectral sensors have hundreds to thousands of very narrow contiguous bands capturing a full spectral signature. Finer spectral resolution enables finer discrimination between surface materials.

> [!question]- Q: What is radiometric resolution and why does it matter?
> A: Radiometric resolution is the number of distinct intensity levels a sensor can record, expressed in bits: 8-bit gives 256 levels, 12-bit gives 4,095, and 16-bit gives 65,536. Higher radiometric resolution allows the sensor to distinguish subtle differences in reflectance, which is important for vegetation indices and water quality analysis.


## Run-length Encoding

> [!question]- Q: What is Run-length Encoding (RLE)?
> A: RLE is a simple lossless compression technique that stores repeated values as a single value plus a count, rather than listing each instance individually (e.g., AAAAAABBBCC becomes 6A 3B 2C).

> [!question]- Q: When does RLE compress effectively?
> A: RLE works best when data has long runs of the same value, such as black-and-white images, binary masks (cloud masks, water masks), and land cover rasters where large areas share the same class value.

> [!question]- Q: How is RLE used in GIS raster formats?
> A: In GIS, RLE (as PackBits) is one of the compression options supported by TIFF and GeoTIFF. It is particularly effective for land cover classification rasters, binary masks, and sparse data with many nodata values. It is also a building block inside more sophisticated schemes like JPEG and PNG.


## S2 Geometry

> [!question]- Q: What is S2 Geometry?
> A: S2 is a library and Discrete Global Grid System (DGGS) developed by Google that represents and indexes geographic regions on a sphere using a hierarchical grid of cells derived by projecting the sphere onto the faces of a cube.

> [!question]- Q: How does S2 index spatial data, and why is that significant?
> A: S2 assigns each cell a unique 64-bit integer ID that encodes both position and hierarchy level. This allows spatial data to be indexed in a regular B-Tree without needing specialized spatial indexes like R-Trees or GiST, making it efficient for large-scale systems like Google Maps and BigQuery.

> [!question]- Q: What is the cell hierarchy in S2 Geometry?
> A: S2 has 30 levels of subdivision. Level 0 represents one of six cube faces; Level 30 cells cover approximately 1 cm². Each cell has exactly four children, and cells at the same level are roughly equal in area.

> [!question]- Q: What space-filling curve property does S2 exploit?
> A: S2 uses a Hilbert curve (six linked Hilbert curves forming one continuous loop over the sphere), which means nearby cells have nearby integer IDs. This locality-preserving property enables efficient proximity and range queries on standard database indexes.


## SatCLIP

> [!question]- Q: What is SatCLIP?
> A: SatCLIP is a geospatial location encoder that learns global representations of geographic locations from satellite imagery using a CLIP-style contrastive learning framework, producing embeddings that capture environmental and ecological context of a location.

> [!question]- Q: What problem does SatCLIP solve?
> A: Many geospatial ML tasks require understanding the characteristics of a location (climate, ecology, land cover) without having task-specific labels. SatCLIP provides rich pre-trained location embeddings derived from satellite imagery that can be used as features for downstream prediction tasks.

> [!question]- Q: How does SatCLIP differ from RemoteCLIP?
> A: RemoteCLIP aligns image embeddings with text descriptions for scene understanding tasks. SatCLIP focuses on encoding geographic locations themselves, producing location embeddings that represent the persistent environmental characteristics of a place rather than describing individual image scenes.


## Satellite

> [!question]- Q: What is the difference between a taskable and a systematic satellite?
> A: A taskable satellite (e.g., Maxar WorldView) is directed to image specific targets on customer request. A systematic satellite (e.g., Planet Doves, Sentinel) images everything in its swath continuously on a fixed schedule without being directed at specific targets.

> [!question]- Q: What is Ground Sample Distance (GSD)?
> A: GSD is the real-world size represented by one pixel, the most precise term for what is loosely called "resolution." For example, WorldView-3's 0.31m GSD means each pixel covers a 31cm × 31cm area on the ground. GSD increases (resolution worsens) at larger off-nadir angles.

> [!question]- Q: What is orthorectification?
> A: Orthorectification is the process of correcting satellite imagery for terrain distortion and sensor geometry so that the image is planimetrically accurate, meaning features appear in their true geographic positions regardless of terrain relief.

> [!question]- Q: What is Analysis Ready Data (ARD)?
> A: ARD is imagery preprocessed to a standard that allows direct use without additional radiometric or geometric correction. Typically ARD is orthorectified, atmospherically corrected to surface reflectance, cloud-masked, and in a consistent coordinate system.

> [!question]- Q: What is pansharpening?
> A: Pansharpening is the process of fusing a high-resolution panchromatic band with lower-resolution multispectral bands to produce a high-resolution color image, combining the spatial detail of the pan band with the spectral information of the multispectral bands.

> [!question]- Q: What is a Sun-Synchronous Orbit and why is it important for EO satellites?
> A: A Sun-Synchronous Orbit (SSO) is a polar LEO orbit where the orbital plane precesses at the same rate Earth orbits the Sun, so the satellite always crosses the equator at the same local solar time. This ensures consistent lighting conditions across all images, which is critical for change detection and time-series analysis.

> [!question]- Q: What is InSAR coherence and what does it indicate?
> A: InSAR coherence measures how similar the radar phase signal is between two SAR acquisitions of the same area. High coherence means the surface has not changed between acquisitions; low coherence (decorrelation) indicates surface change, such as vegetation growth, soil disturbance, or flooding.


## Satellogic

> [!question]- Q: What is Satellogic?
> A: Satellogic is an Earth Observation company, founded in Argentina in 2010, focused on making high-resolution satellite imagery affordable and accessible. It differentiates on low cost and high volume, aiming to commoditize satellite imagery.

> [!question]- Q: What makes Satellogic unusual among commercial EO companies?
> A: Satellogic notably includes hyperspectral sensors (hundreds of spectral bands) on small satellites, which is unusual at their price point. This allows for both multispectral and hyperspectral imaging at sub-meter resolution from a low-cost constellation.

> [!question]- Q: How does Satellogic's business model compare to Maxar or Planet?
> A: Satellogic leads on price disruption: Maxar focuses on highest commercial resolution (dominant in US defense/intelligence), Planet on largest constellation and daily global coverage, while Satellogic differentiates by offering affordable high-resolution and hyperspectral data to a broader market.


## SEN12MS

> [!question]- Q: What is SEN12MS?
> A: SEN12MS is a curated dataset of co-registered Sentinel-1 SAR and Sentinel-2 optical image patch pairs, sampled globally across all four seasons, designed to enable multi-modal and multi-seasonal Earth observation classification research.

> [!question]- Q: What are the key statistics of the SEN12MS dataset?
> A: SEN12MS contains 180,662 triplets of dual-polarization SAR patches, multispectral Sentinel-2 patches, and MODIS land cover maps. All patches are georeferenced at 10m GSD and cover all inhabited continents across all meteorological seasons.

> [!question]- Q: Who created SEN12MS and what tools were used?
> A: SEN12MS was created by researchers exploiting freely available Copernicus program data (Sentinel-1 and Sentinel-2), with cloud computing performed on Google Earth Engine. It was released in June 2019.

> [!question]- Q: What ML tasks is SEN12MS designed to support?
> A: SEN12MS is designed to support deep learning tasks such as scene classification and semantic segmentation for land cover mapping, particularly approaches that leverage multi-modal (optical + SAR) and multi-seasonal data.


## Sentinel

> [!question]- Q: What is the Sentinel program?
> A: Sentinel is a family of satellite missions under the EU Copernicus program run by ESA, each designed for a specific observation type. All data is free and open access, which transformed EO research and industry globally.

> [!question]- Q: What does Sentinel-2 measure and what are its key specs?
> A: Sentinel-2 is a multispectral optical sensor capturing 13 spectral bands (visible, NIR, and SWIR) at resolutions of 10m, 20m, and 60m, with a 290km swath and ~5-day revisit (with both 2A and 2B satellites). It is the workhorse of free EO data for land cover, agriculture, and vegetation monitoring.

> [!question]- Q: What are Sentinel-2's red-edge bands and why are they significant?
> A: Sentinel-2's bands B5, B6, and B7 are red-edge bands particularly sensitive to chlorophyll content and plant stress. This is Sentinel-2's distinct advantage over Landsat and is important for precision agriculture and crop stress detection.

> [!question]- Q: What does Sentinel-1 measure and what are its primary uses?
> A: Sentinel-1 is a C-Band SAR mission with two satellites (1A and 1C) on a 6-day repeat cycle. Its primary uses are surface deformation mapping (InSAR), flood mapping, sea ice monitoring, ship detection, and agricultural monitoring, since SAR sees through clouds and darkness.

> [!question]- Q: What is Sentinel-6 and what does it measure?
> A: Sentinel-6 is a radar altimeter satellite launched in 2020 that measures sea surface height with millimeter precision. Its primary use is sea level rise monitoring and ocean circulation studies.


## Shader

> [!question]- Q: What is a shader?
> A: A shader is a small program that runs on the GPU, written in GLSL (OpenGL Shading Language), a C-like language designed for parallel numeric computation. Originally designed to compute how light shades a surface, shaders are now used for any per-vertex or per-pixel computation.

> [!question]- Q: What is the difference between a vertex shader and a fragment shader?
> A: A vertex shader runs once per vertex, taking coordinates and per-feature data as input and outputting screen positions. A fragment shader runs once per pixel, taking interpolated values from nearby vertices as input and outputting an RGBA color.

> [!question]- Q: Why are shaders important for web map performance?
> A: Shaders run in parallel on the GPU; if you have 10,000 hexagons, the GPU runs 10,000 vertex shader instances simultaneously. This parallelism is why WebGL-based maps are fast, whereas equivalent CPU-based processing would be orders of magnitude slower.


## Short-Wave Infrared

> [!question]- Q: What does SWIR stand for and what wavelength range does it cover?
> A: SWIR stands for Short-Wave Infrared, covering approximately 1,000–2,500 nm. It is commonly split into SWIR-1 and SWIR-2 sub-ranges.

> [!question]- Q: What makes SWIR sensors different from standard optical sensors?
> A: SWIR wavelengths require dedicated sensors (InGaAs or HgCdTe detectors) rather than standard silicon, making them more expensive. SWIR penetrates thin haze and smoke better than visible/NIR bands.

> [!question]- Q: What are the key applications of SWIR in remote sensing?
> A: SWIR is sensitive to liquid water content in vegetation and soil (wet soil appears dark, dry soil bright), differentiates minerals and rock types, and SWIR-2 (~2080–2350 nm) is particularly effective for burned area mapping via the Normalized Burn Ratio (NBR).

> [!question]- Q: Why do Sentinel-2 and Landsat have more analytical power than typical RGB cameras?
> A: Because they include SWIR bands that reveal surface properties invisible to the human eye, such as soil moisture, mineral composition, vegetation water stress, and fire scars, in addition to visible and NIR bands.


## Shuttle Radar Topography Mission

> [!question]- Q: What does SRTM stand for and what did it produce?
> A: SRTM stands for Shuttle Radar Topography Mission. It was an 11-day NASA space shuttle mission in February 2000 that produced the first near-global, high-resolution Digital Elevation Model of Earth's land surface.

> [!question]- Q: What technique did SRTM use to measure elevation?
> A: SRTM used Single-Pass InSAR: two radar antennas separated by a 60-meter boom extended from the shuttle cargo bay. One antenna transmitted radar pulses and both received returns; the phase difference from the geometric separation was converted to elevation with no temporal decorrelation.

> [!question]- Q: What are the SRTM product resolutions?
> A: SRTM1 is 1 arc-second (~30m at the equator), released in 2014. SRTM3 is 3 arc-seconds (~90m), available from the beginning. Both are Digital Surface Models capturing the top of whatever is on the surface, including vegetation canopy.

> [!question]- Q: What has superseded SRTM as the best freely available global DEM?
> A: The Copernicus DEM (derived from TanDEM-X/WorldDEM) is generally considered the best freely available global DEM, with better accuracy than SRTM, more recent data, and better void filling. It was released freely in 2021.


## Simple Features Access

> [!question]- Q: What is Simple Features Access?
> A: Simple Features Access is a widely adopted OGC (Open Geospatial Consortium) standard for storing and accessing 2D vector geospatial data (points, lines, polygons) in databases and GIS software.

> [!question]- Q: What does Simple Features Access define?
> A: It defines geometry types (Point, LineString, Polygon, MultiPoint, MultiCurve, MultiSurface), text (WKT) and binary (WKB) representations for data exchange, and spatial functions for databases prefixed with ST_ (e.g., ST_Intersects, ST_Contains).

> [!question]- Q: Where is Simple Features Access most commonly used?
> A: It is used extensively by spatial databases like PostGIS, as well as most GIS software and geospatial libraries, making it the de facto standard for vector geometry interchange and spatial SQL.


## SkySense

> [!question]- Q: What is SkySense?
> A: SkySense is a billion-scale remote sensing foundation model from Tencent AI, pre-trained on 21.5 million temporal sequences of multi-modal Remote Sensing Imagery (RSI) combining both optical and SAR data.

> [!question]- Q: What makes SkySense architecturally distinctive?
> A: SkySense uses a factorized multi-modal spatiotemporal encoder that takes temporal sequences of optical and SAR data as input. It is pre-trained using Multi-Granularity Contrastive Learning and Geo-Context Prototype Learning to capture representations across different modalities, spatial granularities, and geographic contexts.

> [!question]- Q: How does SkySense perform compared to other remote sensing foundation models?
> A: SkySense outperforms 18 recent remote sensing foundation models across 16 datasets and 7 tasks, surpassing models like GFM, SatLas, and Scale-MAE by margins of 2.76–3.67% on average. It handles tasks from single- to multi-modal, static to temporal, and classification to localization.


## Slew

> [!question]- Q: What is slew rate in the context of satellites?
> A: Slew rate is the speed at which a satellite can rotate (reorient) its sensor from one target to another, typically measured in degrees per second. A higher slew rate allows a satellite to image more targets in a single pass and enables more flexible tasking.

> [!question]- Q: Why does slew rate matter for Very High Resolution satellites?
> A: VHR satellites like Maxar WorldView are taskable and must rotate to point at specific targets off-nadir. A faster slew rate means the satellite can image more targets during a single overpass, directly affecting commercial throughput and revisit flexibility.


## Space-filling Curve

> [!question]- Q: What is a space-filling curve?
> A: A space-filling curve is a path through space that visits every point in a 2D (or higher-dimensional) area while remaining a continuous 1D curve. It translates multi-dimensional data into one-dimensional data while preserving locality.

> [!question]- Q: What is the key property of space-filling curves that makes them useful for spatial indexing?
> A: The locality-preserving property: points that are nearby in 2D space are mapped to nearby positions on the 1D curve. This means spatial data can be indexed in a standard 1D index (like a B-Tree) with nearby cells having nearby IDs, enabling efficient range and proximity queries.

> [!question]- Q: What are the classic examples of space-filling curves used in geospatial systems?
> A: The Hilbert Curve (used in S2 Geometry and H3), the Z-Order (Morton) Curve (used in many spatial databases and GeoParquet), and the Peano Curve. The Hilbert Curve is generally preferred because it has better locality preservation than the Z-Order Curve.

> [!question]- Q: How does S2 Geometry use space-filling curves?
> A: S2 links six Hilbert curves together to form a single continuous loop over the entire sphere, one per face of the cube projection. Cell IDs are assigned along this curve so that spatially nearby cells have numerically nearby IDs, enabling fast range-based spatial lookups.


## SpaceNet

> [!question]- Q: What is SpaceNet?
> A: SpaceNet is not a single dataset but a collection of challenges and datasets released collaboratively by In-Q-Tel, Nvidia, Amazon Web Services, and Maxar starting in 2016, each targeting a different high-resolution satellite imagery mapping problem.

> [!question]- Q: What type of imagery does SpaceNet use?
> A: SpaceNet uses commercial Very High Resolution (VHR) satellite imagery, primarily from Maxar's WorldView constellation.

> [!question]- Q: What were the first SpaceNet challenges focused on?
> A: The first two SpaceNet challenges focused on automated building footprint extraction, and a subsequent challenge focused on road network extraction from satellite imagery.

> [!question]- Q: What is the broader goal of SpaceNet?
> A: SpaceNet aims to accelerate foundational mapping in dynamic scenarios (such as natural disasters) by combining frequent satellite revisits with advanced machine learning techniques, reducing dependence on manual human labeling.


## Spark JS

> [!question]- Q: What is Spark JS?
> A: Spark JS (named to distinguish it from Apache Spark) is an advanced Gaussian Splatting renderer for Three.js, built by World Labs, designed to create, stream, and render large 3D Gaussian Splat worlds on the web on any device.

> [!question]- Q: What is the key innovation in Spark 2.0?
> A: Spark 2.0 introduces a Level-of-Detail (LoD) Splat Tree structure, where any splat file is organized into a tree with all original splats as leaves and interior nodes representing progressively downsampled versions. At render time, the system picks the best set of splats for the current viewpoint and distance.

> [!question]- Q: What file format does Spark define for streaming splats?
> A: Spark defines the .RAD (RADiance) file format, an extensible format that stores a precomputed LoD splat tree and enables streaming arbitrary chunks of splats via HTTP range requests, analogous to how Cloud-Optimized GeoTIFF uses range requests for imagery.

> [!question]- Q: How does Spark 2.0 handle memory management for huge scenes?
> A: Spark 2.0 implements a shared LRU "splat page table" modeled after OS virtual memory paging, pre-allocating a fixed GPU memory pool shared across all splat objects and automatically fetching and evicting chunks based on the current viewpoint, enabling scenes with 100M–1B+ splats to render in real time on mobile devices.


## Spatial ETL

> [!question]- Q: What is Spatial ETL?
> A: Spatial ETL is the geospatial version of a standard Extract-Transform-Load data pipeline, with the added complexity of geometries, projections, spatial operations, and often very large file sizes.

> [!question]- Q: What is the most common transform step in a spatial ETL pipeline?
> A: Projection handling is almost always necessary, as source data arrives in inconsistent CRSs and must be reprojected to a common system before any spatial operation. Web Mercator (EPSG:3857) is used for display, a local UTM zone for accurate distance/area calculations, and WGS84 for storage.

> [!question]- Q: Why is geometry fixing important in spatial ETL?
> A: Real-world vector data is dirty: self-intersecting polygons, unclosed rings, duplicate vertices, slivers, and wrong winding order are common. Skipping geometry validation causes silent failures in downstream spatial operations.

> [!question]- Q: Why are S2 or H3 cell IDs commonly used as partitioning keys in large-scale spatial ETL?
> A: Spatial data cannot be partitioned simply by row ID. Partitioning by geography ensures workers processing adjacent tiles don't constantly need data from each other's partitions. S2 and H3 cell IDs encode geographic location and hierarchy, making them natural partition keys.

> [!question]- Q: What is Tippecanoe's role in a spatial ETL pipeline?
> A: Tippecanoe converts processed vector datasets into vector tile sets (MBTiles, PMTiles) for web visualization, deciding what features to show at each zoom level by simplifying geometries, dropping features, and clustering points to keep tile sizes manageable.


## SpatioTemporal Asset Catalog

> [!question]- Q: What does STAC stand for and what problem does it solve?
> A: STAC stands for SpatioTemporal Asset Catalog. It solves the problem of every satellite data provider having their own metadata format and discovery mechanism by providing a standard JSON specification for describing and searching geospatial assets.

> [!question]- Q: What are the four components of STAC?
> A: An Item (describes one asset, e.g., one Landsat scene), a Collection (groups related items, e.g., all Sentinel-2 L2A scenes), a Catalog (a top-level container linking to collections), and an API (a standard REST search endpoint supporting filtering by bounding box, datetime, cloud cover, and collections).

> [!question]- Q: How does STAC pair with Cloud-Optimized GeoTIFF?
> A: STAC+COG is the standard stack for satellite imagery. STAC tells you what exists and where the files are; COG lets you efficiently read only the spatial subset you need via HTTP range requests. Together they enable cloud-native geospatial workflows without downloading full files.

> [!question]- Q: What are the major public STAC catalogs?
> A: Microsoft Planetary Computer and Element84 Earth Search are the major public STAC catalogs, hosting large collections of Sentinel-2, Landsat, and other EO datasets.


## Spectral Index

> [!question]- Q: What is a spectral index?
> A: A spectral index is any mathematical combination of spectral bands designed to highlight a specific surface property. Common examples include vegetation indices (NDVI), water indices (NDWI), burn indices (NBR), built-up indices (NDBI), and soil indices (BSI).

> [!question]- Q: What are the main categories of spectral indices used in remote sensing?
> A: Vegetation (e.g., NDVI — plant biomass/health), Water (e.g., NDWI — surface water and moisture), Burn (e.g., NBR — fire scars), Built-up (e.g., NDBI — urban surfaces), and Soil (e.g., BSI — bare soil and organic carbon).

> [!question]- Q: Why should spectral indices be computed from Surface Reflectance rather than Top-of-Atmosphere data?
> A: Surface Reflectance removes atmospheric effects, so the same index computed on different dates or from different sensors gives comparable results. TOA data includes variable atmospheric scattering and absorption, meaning index values would differ even if the surface hadn't changed.


## Splat

> [!question]- Q: What is a splat in the context of point cloud visualization?
> A: A splat is a rendering technique for LiDAR point clouds where each mathematically dimensionless point (just an XYZ coordinate) is rendered as a small screen-space disc or circle sized to fill the gaps between neighboring points, making the result look like a continuous surface rather than sparse dots.

> [!question]- Q: How is disc size determined in splat rendering?
> A: Disc size is typically computed from the local point spacing and camera distance, so the splat fills the expected gap between points at the current zoom level without overlapping excessively.

> [!question]- Q: How does splat rendering relate to Gaussian Splatting?
> A: Traditional splats are flat discs for point cloud visualization. Gaussian Splatting (3DGS) extends this concept to volumetric 3D Gaussians with orientation, opacity, and color, enabling full scene reconstruction and novel view synthesis rather than just point cloud display.


## SPOT

> [!question]- Q: What does SPOT stand for?
> A: SPOT stands for Satellite Pour l'Observation de la Terre (French for "Satellite for Earth Observation"), a French commercial high-resolution optical Earth observation satellite system run by Spot Image, based in Toulouse, France.

> [!question]- Q: What type of orbit do SPOT satellites use?
> A: SPOT satellites use a circular, polar, Sun-synchronous orbit at 832km altitude with 98.7-degree inclination, allowing them to fly over any point on Earth within 26 days.

> [!question]- Q: What are the specifications of SPOT 6, the currently active satellite?
> A: SPOT 6 (launched 2012) captures panchromatic imagery at 1.5m resolution, multispectral imagery at 6m resolution across blue, green, red, and NIR bands, with a 60km × 60km footprint.

> [!question]- Q: How many SPOT satellites have there been and what is the current status?
> A: There have been 7 SPOT satellites (SPOT 1–7). SPOT 6 is still active; all others have been decommissioned or stopped functioning. SPOT 7 stopped functioning and was decommissioned in 2023.


## Sun-Synchronous Orbit

> [!question]- Q: What is a Sun-Synchronous Orbit (SSO)?
> A: An SSO is a Low Earth Orbit where the orbital plane precesses at exactly the same rate that Earth orbits the Sun (~1 degree per day), so the satellite always crosses the equator at the same local solar time on every pass.

> [!question]- Q: Why is Sun-Synchronous Orbit the standard for optical Earth observation?
> A: SSO ensures consistent solar illumination conditions across all images taken at any time of year, which is critical for change detection (you're not confusing shadow changes with land-cover changes) and time-series analysis.

> [!question]- Q: What orbital parameters define a Sun-Synchronous Orbit?
> A: Altitude of ~400–1000 km (a subset of LEO) and inclination of ~97–98 degrees (slightly retrograde). The slightly retrograde inclination causes the orbital plane to precess in sync with the Sun due to Earth's equatorial bulge.

> [!question]- Q: Which major EO satellites use Sun-Synchronous Orbit?
> A: Nearly all optical Earth-observation satellites use SSO, including Sentinel-2, Landsat, SPOT, and PlanetScope. It is the default orbit for remote sensing.


## Surface Reflectance

> [!question]- Q: What is Surface Reflectance (SR)?
> A: Surface Reflectance is the fraction of incoming sunlight actually reflected by the Earth's surface itself after removing the effects of the atmosphere. It represents the true spectral signature of the surface, as opposed to the atmosphere-contaminated Top-of-Atmosphere reflectance.

> [!question]- Q: Why does Surface Reflectance matter for remote sensing analysis?
> A: SR enables reliable change detection across different dates, meaningful spectral index computation (NDVI, NDWI, NBR all assume SR), multi-sensor data fusion (combining Landsat and Sentinel data), and ML models that generalize across scenes rather than learning atmospheric artifacts.

> [!question]- Q: What is atmospheric correction and why is it needed to derive Surface Reflectance?
> A: Atmospheric correction models what the atmosphere did to the light and inverts it, removing path radiance (atmospheric scattering directly into the sensor), adjacency effects, and aerosol/water vapor absorption. It requires knowledge of Aerosol Optical Depth, water vapor, ozone, and solar/view geometry at acquisition time.

> [!question]- Q: What is the tradeoff of using Surface Reflectance vs Top-of-Atmosphere data?
> A: SR requires accurate atmospheric data at the time of acquisition. If that auxiliary data is unavailable or low quality, the correction can introduce its own errors, making TOA potentially more reliable in some cases.


## Swath

> [!question]- Q: What is a satellite swath?
> A: The swath is the strip of the Earth's surface actually imaged by the sensor as the satellite moves along its ground track. Its width is determined by the sensor's field of view and pointing angle.

> [!question]- Q: What is the relationship between swath width and resolution?
> A: There is a fundamental tradeoff: a wider swath means more area is covered per pass, but typically at lower spatial resolution. Narrowing the field of view to achieve higher resolution reduces swath width and therefore temporal resolution (longer time between revisits of the same area).

> [!question]- Q: How does the swath relate to the ground track?
> A: The ground track is the path the satellite traces over Earth's surface. The swath is the strip actually imaged, which is normally to the side of (or centered on) the ground track. If the satellite is tasked off-nadir, the swath may not even overlap the ground track.


## Synthetic Aperture Radar

> [!question]- Q: What does SAR stand for and how does it work?
> A: SAR stands for Synthetic Aperture Radar. A satellite emits microwave pulses toward the ground and records the backscattered energy. By combining measurements taken from many slightly different positions along the flight path, it mathematically synthesizes a very large antenna aperture to produce a sharp image.

> [!question]- Q: What are the key advantages of SAR over optical remote sensing?
> A: SAR sees through clouds and darkness (it does not depend on sunlight), detects surface roughness and moisture from backscatter, enables millimeter-scale surface deformation measurement via InSAR, and encodes structural properties (roughness, geometry, dielectric properties) rather than just color.

> [!question]- Q: How does SAR wavelength (band) determine what it can observe?
> A: Shorter wavelengths (X-Band ~3cm) bounce off surfaces for fine detail of urban areas and sea ice. Longer wavelengths penetrate deeper: C-Band (~5.6cm) is good for surface change and agriculture, L-Band (~24cm) penetrates forest canopy for biomass and soil, and P-Band (~70cm) penetrates deep into dense forest for root-zone and biomass estimation.

> [!question]- Q: What is the difference between the four SAR imaging modes?
> A: Stripmap images a continuous fixed-width strip (default). ScanSAR sweeps across multiple strips for wider coverage at coarser resolution. Spotlight dwells on one small area for finer resolution but no swath. TOPSAR is a hybrid of ScanSAR that fixes its scalloping artifacts and is compatible with InSAR.

> [!question]- Q: What is SAR polarization and why does it matter?
> A: SAR can transmit and receive in horizontal (H) or vertical (V) polarization. Co-pol (HH, VV) responds to surface and double-bounce scattering (bare soil, buildings), while cross-pol (HV, VH) responds to volume scattering (vegetation canopy). Different polarization combinations reveal different surface properties.

> [!question]- Q: What is InSAR and what can it measure?
> A: InSAR (Interferometric SAR) compares the phase of two SAR acquisitions over the same area to measure surface deformation at centimeter or millimeter scale. Applications include earthquake displacement mapping, volcano inflation monitoring, glacier movement tracking, and ground subsidence detection.


## Tagged Image File Format

> [!question]- Q: What does TIFF stand for and what makes it distinctive?
> A: TIFF stands for Tagged Image File Format. Its key feature is a tag-based structure: rather than a fixed layout, a TIFF file is a collection of metadata tags describing what the data is and how it's stored, making it highly flexible.

> [!question]- Q: What types of data can a TIFF file contain?
> A: TIFF supports many data types (uint8, uint16, float32, etc.), multiple bands, multiple compression schemes (LZW, DEFLATE, PackBits, JPEG), and multiple images (pages) in one file. This flexibility makes it the foundation for geospatial formats like GeoTIFF.

> [!question]- Q: What is the relationship between TIFF and GeoTIFF?
> A: GeoTIFF is an extension of TIFF that adds geospatial metadata tags (coordinate reference system, geotransform, projection parameters), allowing the image to be located precisely on the Earth's surface. It is the dominant format for raster geospatial data.


## TanDEM-X

> [!question]- Q: What does TanDEM-X stand for?
> A: TanDEM-X stands for "TerraSAR-X add-on for Digital Elevation Model measurements." It is a German radar mission operated in partnership with Airbus.

> [!question]- Q: How does TanDEM-X collect elevation data?
> A: TanDEM-X flies in close formation with TerraSAR-X, separated by only a few hundred meters to a few kilometers, forming a single-pass InSAR system. The two antennas at slightly different positions at the same moment give coherent interferometric phase measurements without temporal decorrelation.

> [!question]- Q: What is TanDEM-X's primary product and its specifications?
> A: The primary product is the WorldDEM: a global DEM at 12m resolution with ~2m relative vertical accuracy and ~10m absolute vertical accuracy. This is significantly better than SRTM in both resolution and accuracy, and is believed to be the highest-resolution global DEM available.

> [!question]- Q: What is the freely available product derived from TanDEM-X data?
> A: The Copernicus DEM (GLO-30) is a freely available 30m DEM (and GLO-90 at 90m) derived from WorldDEM, released in 2021 and generally considered the best freely available global DEM.


## Task

> [!question]- Q: What is satellite tasking?
> A: Tasking is the process of instructing a satellite to acquire imagery of a specific location at a specific time. Commercial taskable satellites like Maxar's WorldView can be directed to specific targets on customer request, contrasted with systematic satellites that image everything in their swath continuously.

> [!question]- Q: What is the difference between tasked and systematic satellite acquisition modes?
> A: In tasked mode, a customer or operator directs the satellite to a specific target (how Maxar WorldView operates, with competing customer requests and priority queuing). In systematic mode, the satellite collects everything in its swath continuously on a fixed schedule regardless of specific requests (how Sentinel-1 and Sentinel-2 operate).


## TerraSAR-X

> [!question]- Q: What is TerraSAR-X?
> A: TerraSAR-X is a German X-Band SAR Earth observation satellite launched in 2007, operated as a public-private partnership between the German Aerospace Center (DLR) and EADS Astrium (now Airbus). It operates from a sun-synchronous polar orbit at 514km altitude.

> [!question]- Q: What is the relationship between TerraSAR-X and TanDEM-X?
> A: TerraSAR-X and its twin satellite TanDEM-X (launched 2010) fly in close formation to form a single-pass InSAR system. Together they acquired the data for the WorldDEM global DEM. TerraSAR-X was the first of the pair; TanDEM-X was designed specifically to fly alongside it.

> [!question]- Q: What SAR band does TerraSAR-X use and what is it good for?
> A: TerraSAR-X uses X-Band (~3cm wavelength), which is best for fine detail of urban areas, sea ice surfaces, and vessel detection. X-Band provides high spatial resolution but limited penetration, so it responds primarily to surface features.


## TESSERA

> [!question]- Q: What is TESSERA?
> A: TESSERA is a pixel-wise foundation model for multi-modal (Sentinel-1 and Sentinel-2) Earth Observation time series that learns robust, label-efficient embeddings. It was released in June 2025.

> [!question]- Q: What training approach does TESSERA use?
> A: TESSERA uses Barlow Twins self-supervised learning with sparse random temporal sampling to enforce invariance to the selection of valid observations (handling irregular time series from cloud cover). Key regularizers include global shuffling (to decorrelate spatial neighborhoods) and mix-based regulation (for robustness under extreme sparsity).

> [!question]- Q: What does TESSERA release publicly?
> A: TESSERA releases global, annual, 10m, pixel-wise int8 embeddings covering the full Earth, together with open model weights, code, and lightweight adaptation heads for large-scale retrieval and inference at planetary scale.

> [!question]- Q: What is the significance of TESSERA's label efficiency?
> A: TESSERA embeddings achieve state-of-the-art accuracy on diverse classification, segmentation, and regression tasks while requiring only a small task head and minimal labeled data. This makes sophisticated EO analysis accessible without large labeled training datasets.


## Thermal Infrared

> [!question]- Q: What does TIR stand for and what does it measure?
> A: TIR stands for Thermal Infrared, covering ~8,000–14,000 nm (8–14 µm). Unlike NIR and SWIR which measure reflected sunlight, TIR measures emitted thermal radiation — heat that objects radiate based on their own temperature.

> [!question]- Q: What are common applications of Thermal Infrared remote sensing?
> A: TIR is used for Land Surface Temperature (LST) estimation, urban heat island mapping (asphalt and concrete emit strongly), vegetation stress analysis (vegetated surfaces stay cooler via evapotranspiration), and sea surface temperature measurement.

> [!question]- Q: Why are TIR sensors lower resolution and more expensive than optical sensors?
> A: TIR sensors must be cooled to detect faint thermal emissions from the ground above their own sensor noise, making them heavier, more expensive, and generally of lower resolution. For example, Landsat's thermal band is 100m resolution compared to 30m for its optical bands.

> [!question]- Q: Which major satellite constellations include Thermal Infrared bands?
> A: Landsat (all versions), ASTER, and ECOSTRESS include TIR bands. Notably, Sentinel-2 has no thermal band, so Landsat or specialized sensors are required for surface temperature work.


## Tile Pyramid

> [!question]- Q: What is a Tile Pyramid?
> A: A tile pyramid is a set of pre-computed versions of raster or vector data at multiple zoom levels. In web mapping, it is the full set of XYZ tiles stored in formats like MBTiles or PMTiles. In raster contexts, it refers to overview levels (image pyramids) embedded in a GeoTIFF.

> [!question]- Q: Why are tile pyramids important for web map performance?
> A: Pre-computing lower-resolution versions means the server or client can serve the appropriate resolution level for the current zoom without computationally expensive on-the-fly downsampling, enabling fast rendering at any zoom level.

> [!question]- Q: What is the difference between a tile pyramid in web mapping vs. a COG overview?
> A: They represent the same underlying idea of pre-computed multi-resolution representations, but the web mapping tile pyramid stores tiles in discrete XYZ coordinates (zoom/x/y) while COG overviews are embedded byte ranges within a single GeoTIFF file, both avoiding the need to downsample on-the-fly.


## Tippecanoe

> [!question]- Q: What is Tippecanoe?
> A: Tippecanoe is an open-source GIS command-line tool originally built by Mapbox (now maintained by Felt) that builds vector tile sets (MBTiles, PMTiles) from large GeoJSON, Geobuf, or CSV datasets.

> [!question]- Q: What is Tippecanoe's core function?
> A: Tippecanoe decides what features to show at each zoom level, simplifying geometries, dropping features, and clustering points so that tiles don't become too large to transmit and render efficiently at low zoom levels.

> [!question]- Q: What is Tippecanoe's design philosophy?
> A: Its goal is to enable a scale-independent view of data so that at any zoom level — from the entire world to a single building — you can see the density and texture of the data rather than a simplification that drops "unimportant" features.

> [!question]- Q: What is the relationship between Tippecanoe and Mapbox Tiling Service?
> A: Mapbox Tiling Service (MTS) is Mapbox's cloud-hosted pipeline that does essentially the same thing as Tippecanoe but runs on Mapbox's infrastructure. Tippecanoe is the open-source equivalent; Mapbox transferred its maintenance to Felt.


## Tobler's First Law of Geography

> [!question]- Q: What is Tobler's First Law of Geography?
> A: "Everything is related to everything else, but near things are more related than distant things." — Waldo Tobler, 1970. It is the foundational assumption underlying almost all spatial statistics.

> [!question]- Q: What is spatial autocorrelation and how does it relate to Tobler's First Law?
> A: Spatial autocorrelation is the formal statistical measurement of how much a variable's values at one location are correlated with values at nearby locations. It quantifies the principle stated in Tobler's First Law: that geographic proximity implies some degree of similarity.

> [!question]- Q: What are the practical implications of Tobler's First Law for machine learning with geospatial data?
> A: Because nearby locations are similar, random train/test splits violate independence assumptions — training samples near test samples cause data leakage. Geospatial ML requires spatially-aware splits (e.g., holding out entire geographic regions) to get honest generalization estimates.


## Top-of-Atmosphere

> [!question]- Q: What does TOA stand for in remote sensing?
> A: TOA stands for Top-of-Atmosphere reflectance. It is the raw radiance captured by a satellite sensor with minimal correction, including atmospheric scattering and absorption effects from aerosols, water vapor, and ozone.

> [!question]- Q: How does Top-of-Atmosphere differ from Surface Reflectance?
> A: TOA includes the distorting effects of the atmosphere, so the same surface can look different on different days due to varying atmospheric conditions. Surface Reflectance removes these effects via atmospheric correction to reveal the true reflectance of the surface itself.

> [!question]- Q: When might TOA data be preferable to Surface Reflectance?
> A: If accurate atmospheric data at the time of acquisition is unavailable or low quality, atmospheric correction can introduce its own errors, making TOA actually more reliable. Some applications like cloud detection also work directly on TOA data.


## TorchGeo

> [!question]- Q: What is TorchGeo?
> A: TorchGeo is PyTorch's official geospatial deep learning library from Microsoft Research, analogous to torchvision but built for remote sensing and Earth observation tasks.

> [!question]- Q: What specific problems does TorchGeo solve for EO machine learning?
> A: Raw geospatial data is hostile to standard ML pipelines because imagery has arbitrary CRSs, resolution, and extent (you can't just stack tensors), multi-sensor fusion requires spatial alignment, and EO datasets have spatial autocorrelation that breaks random train/test splits.

> [!question]- Q: What does TorchGeo provide out of the box?
> A: TorchGeo provides 50+ ready-to-use EO datasets (Sentinel, Landsat, NAIP, BigEarthNet, etc.) with automatic download and CRS normalization, geospatial-aware samplers that respect spatial boundaries, transforms for arbitrary band counts and spectral index computation, and pretrained EO model weights including foundation models like Clay.


## United States Geological Survey

> [!question]- Q: What does USGS stand for and what is the USGS National Map?
> A: USGS stands for United States Geological Survey. The USGS National Map (nationalmap.gov) is USGS's primary platform for distributing authoritative US geospatial data including elevation, hydrology, land cover, transportation, and orthoimagery.

> [!question]- Q: What key datasets does the USGS National Map contain?
> A: It contains 3DEP elevation data, the National Hydrography Dataset (NHD) for water networks, the Watershed Boundary Dataset (WBD), the National Land Cover Database (NLCD) at 30m, National Structures Dataset (buildings and landmarks), transportation networks, and official place names (GNIS).

> [!question]- Q: What is the NLCD and how often is it updated?
> A: The National Land Cover Database (NLCD) is a 30m resolution land cover classification of the contiguous United States that is updated approximately every few years, providing a consistent time series of land cover change at national scale.


## Universal Polar Stereographic

> [!question]- Q: What does UPS stand for?
> A: UPS stands for Universal Polar Stereographic.

> [!question]- Q: What is the Universal Polar Stereographic projection used for?
> A: UPS is a conformal map projection used to cover the polar regions (above 84°N and below 80°S) that are not covered by the Universal Transverse Mercator (UTM) system. Together, UTM and UPS provide complete global coverage for military and surveying applications.

> [!question]- Q: How does UPS relate to UTM?
> A: UTM covers the Earth between 80°S and 84°N in 60 longitudinal zones. UPS covers the remaining polar caps beyond those latitudes using a stereographic projection centered on each pole, complementing UTM to form the complete Military Grid Reference System (MGRS).


## Universal Transverse Mercator

> [!question]- Q: What does UTM stand for?
> A: UTM stands for Universal Transverse Mercator.

> [!question]- Q: How is the Earth divided in the UTM system?
> A: The Earth is divided into 60 zones, each 6 degrees of longitude wide, numbered 1–60 starting at the International Date Line (180°W) moving east. Each zone uses a Transverse Mercator projection with meters for northings (from the equator) and eastings (from a 500,000m central meridian).

> [!question]- Q: What is the main advantage of using UTM for geospatial analysis?
> A: UTM zones provide a planar coordinate system in meters, enabling accurate distance and area calculations without the distortions introduced by geographic coordinates (degrees). Each zone minimizes distortion within its 6-degree band, making it suitable for local to regional analysis.

> [!question]- Q: What are the limitations of UTM?
> A: UTM only covers the Earth between 80°S and 84°N (the polar regions require UPS). Because there are 60 separate zones, data spanning multiple zones must be reprojected into a common zone or a different projection, and accuracy degrades toward the edges of each zone.


## Van Allen Radiation Belt

> [!question]- Q: What are the Van Allen Radiation Belts?
> A: The Van Allen Radiation Belts are two zones of energetic charged particles (primarily electrons and protons) trapped by Earth's magnetic field, surrounding the Earth like donuts. The inner belt is roughly 1,000–12,000 km altitude and the outer belt 13,000–60,000 km.

> [!question]- Q: Why are the Van Allen Belts significant for satellite operations?
> A: Satellites passing through the Van Allen Belts are exposed to intense radiation that can damage electronics, degrade solar panels, corrupt data, and cause single-event upsets in computer memory. Satellite designs must account for radiation hardening if they operate in or transit through these regions.

> [!question]- Q: Which satellite orbits are affected by the Van Allen Belts?
> A: Low Earth Orbit (LEO, below ~1,000 km) satellites generally avoid the main belts. Geostationary orbit (GEO, ~36,000 km) satellites sit in or just outside the outer belt. Medium Earth Orbit (MEO) satellites like GPS pass through the belts, requiring radiation-hardened components.


## Vector Tile

> [!question]- Q: What is a Vector Tile?
> A: A vector tile is a pre-packaged, binary-encoded chunk of geospatial vector data (points, lines, polygons with attributes) for a specific geographic tile at a specific zoom level, typically encoded in the Mapbox Vector Tile (MVT) format using Protocol Buffers.

> [!question]- Q: How do vector tiles differ from raster tiles?
> A: Raster tiles are pre-rendered images (PNG/JPEG) where styling is baked in. Vector tiles contain the raw geometry and attribute data, so styling is applied client-side at render time. This allows dynamic styling, filtering, and interaction without regenerating tiles.

> [!question]- Q: What are the advantages of vector tiles for web mapping?
> A: Vector tiles enable client-side dynamic styling (change colors, show/hide layers without new server requests), interactive feature queries (click a building to see its attributes), smooth zoom transitions, and much smaller file sizes than equivalent raster tiles for many data types.

> [!question]- Q: How does Tippecanoe relate to vector tiles?
> A: Tippecanoe is the standard open-source tool for generating vector tile sets from large GeoJSON or CSV datasets, handling the complexity of deciding which features to include, simplify, or drop at each zoom level.


## Vegetation Index

> [!question]- Q: What is a vegetation index?
> A: A vegetation index is a spectral index computed from satellite bands specifically targeting photosynthetic activity, biomass, canopy structure, or plant stress. They typically exploit chlorophyll's strong absorption in red wavelengths and high reflectance in NIR.

> [!question]- Q: What are common vegetation indices and what do they measure?
> A: NDVI (Normalized Difference Vegetation Index) measures general plant biomass and health. EVI corrects for soil and atmospheric effects, better for dense canopy. SAVI adjusts NDVI for sparse vegetation over bright soil. NDRE uses the red-edge band for crop stress detection. LAI (Leaf Area Index) is a physically-grounded measure of canopy density.

> [!question]- Q: Why is NDVI less accurate in dense canopy or bright soil conditions?
> A: NDVI saturates at high vegetation densities (dense forest all looks the same) and is affected by soil brightness in sparse vegetation cover. EVI and SAVI respectively address these limitations by incorporating blue-band and soil adjustment factors.


## Very High Resolution

> [!question]- Q: What is Very High Resolution (VHR) imagery?
> A: VHR is a general term for satellite imagery at sub-meter resolution, typically from commercial operators like Maxar and Planet (SkySat). Data volume increases quadratically with resolution, and VHR satellites typically image a smaller swath.

> [!question]- Q: Why is time-series analysis difficult with VHR imagery?
> A: VHR satellites do agile pointing (slewing), so between acquisitions the off-nadir angle and direction may vary significantly (e.g., 30° from one direction, then 25° from another). This causes pixel-level misalignment (5–10 pixels of shift per timestep) even after orthorectification, making time-series comparison unreliable.

> [!question]- Q: What are VHR imagery most commonly used for in practice?
> A: VHR is mostly used for image interpretation and visual inspection tasks (port activity, ship detection, parking lots, building footprints) rather than time-series analysis, because the precise pixel alignment needed for multi-temporal analysis is difficult to achieve.

> [!question]- Q: What is the tradeoff between VHR and moderate-resolution (e.g., Sentinel-2) for temporal analysis?
> A: Sentinel-2's 10m pixels tolerate the meter-scale positional errors typical of satellite data, making time-series methods work well. VHR's 50cm pixels mean even small positional errors represent many pixels of misalignment, effectively breaking pixel-level temporal analysis.


## Wherobots

> [!question]- Q: What is Wherobots?
> A: Wherobots is a geospatial data analytics company that provides a cloud-native spatial computing platform built on Apache Sedona (distributed spatial SQL on Spark), offering managed infrastructure for large-scale geospatial data processing and EO inference.

> [!question]- Q: What is the relationship between Wherobots and Apache Sedona?
> A: Wherobots was founded by the creators of Apache Sedona, the open-source distributed spatial computing framework. WherobotsDB is their managed, cloud-hosted product built on Apache Sedona, providing spatial SQL at scale without requiring users to manage Spark infrastructure.

> [!question]- Q: What is Wherobots' RasterFlow product?
> A: RasterFlow is Wherobots' serverless EO inference engine that automates the full pipeline from raw satellite imagery to actionable geospatial insights, running PyTorch models at planetary scale and writing results as Apache Iceberg tables.


## Whitebox Tools

> [!question]- Q: What is Whitebox Tools?
> A: Whitebox Tools is a Python library for raster analytics in a notebook/Python setting. It is written in Rust with Python bindings, providing strong capabilities for hydrological analysis, terrain attributes, LiDAR processing, and geomorphometric analysis.

> [!question]- Q: What are Whitebox Tools' core strengths?
> A: Whitebox Tools excels at hydrological analysis from LiDAR-derived DEMs, terrain attributes (slope, aspect, curvature, hillshade), LiDAR ground filtering, DEM void filling, and geomorphometric analysis.

> [!question]- Q: Why is Rust used as the implementation language for Whitebox Tools?
> A: Rust provides memory safety without garbage collection and performance comparable to C/C++, making it well-suited for computationally intensive raster analytics. The Python bindings allow it to be used conveniently in data science workflows.


## Winding Order

> [!question]- Q: What is winding order in geospatial data?
> A: Winding order is the direction (clockwise or counterclockwise) in which the vertices of a polygon ring are listed. It determines which side of a polygon boundary is the "inside" and is significant for correct polygon rendering and spatial operations.

> [!question]- Q: What is the right-hand rule for winding order?
> A: The standard convention in most geospatial contexts: exterior rings (outer boundaries) are listed counterclockwise (CCW), and interior rings (holes) are listed clockwise (CW). If you curl the fingers of your right hand in the direction of vertex traversal, your thumb points toward the "inside."

> [!question]- Q: Why does winding order cause bugs, and which formats disagree?
> A: Different standards use opposite conventions: GeoJSON (RFC 7946) uses CCW exterior rings, while Shapefiles use CW exterior rings — the opposite. WKT/WKB is technically unspecified. This causes silent errors when converting between formats. OpenGL/WebGL use CCW as front-facing, affecting 3D rendering.


## WorldDEM

> [!question]- Q: What is WorldDEM?
> A: WorldDEM is a global Digital Elevation Model at 12m resolution with ~2m relative vertical accuracy and ~10m absolute vertical accuracy, created by the TanDEM-X mission (satellites TerraSAR-X and TanDEM-X) using single-pass InSAR since 2010.

> [!question]- Q: Who sells WorldDEM and what is it used for?
> A: WorldDEM is a commercial product sold by Airbus at per-square-kilometer pricing. It is used for defense, infrastructure planning, telecommunications (radio propagation modeling), aviation, and high-end scientific applications requiring precise terrain data.

> [!question]- Q: What is the relationship between WorldDEM and the Copernicus DEM?
> A: The Copernicus DEM (GLO-30 at 30m, GLO-90 at 90m) is a freely available DEM derived from WorldDEM. It is a processed, edited, and hydrologically conditioned version of the TanDEM-X data, downsampled from the native 12m resolution.


## Xarray

> [!question]- Q: What is Xarray?
> A: Xarray is a Python library for working with labeled multidimensional arrays. It extends NumPy by adding named dimensions and coordinate labels, making it the primary tool for scientific array data in climate, oceanography, atmospheric science, and increasingly Earth observation.

> [!question]- Q: What are the two core data structures in Xarray?
> A: A DataArray is a single labeled N-dimensional array with named dimensions (e.g., time, lat, lon) and coordinates. A Dataset is a collection of DataArrays sharing the same dimensions, equivalent to a NetCDF file in memory.

> [!question]- Q: How is Xarray analogous to Pandas?
> A: Just as Pandas adds labeled rows and columns to NumPy arrays for tabular data, Xarray adds named dimensions and coordinate labels to NumPy arrays for N-dimensional scientific data. Xarray uses Zarr as a storage backend the same way Pandas uses Parquet.

> [!question]- Q: What is the common use of Xarray in Earth observation?
> A: Xarray is almost always used alongside Zarr for working with large EO data cubes — stacks of satellite imagery over time across a region — allowing users to select data by coordinate ranges (e.g., "give me the Pacific Northwest for July 2023") without loading the entire dataset.


## Zarr

> [!question]- Q: What is Zarr?
> A: Zarr is a chunk-based, compressed, N-dimensional array format designed from the ground up for cloud object storage. It stores each chunk as a separate file/object rather than as a byte range inside a monolithic file, enabling parallel reads and writes directly from object storage like S3.

> [!question]- Q: How does Zarr differ from Cloud-Optimized GeoTIFF (COG)?
> A: COG keeps everything in one file and uses HTTP range requests to fetch specific byte ranges — simple, one URL to share. Zarr is genuinely multi-file with potentially millions of objects in S3, optimized for parallel access to large N-dimensional arrays. COG is best for individual 2D scenes; Zarr is best for large multi-temporal data cubes.

> [!question]- Q: What is Zarr's chunk addressing scheme?
> A: Zarr chunk locations are implicit from their filename (chunk 2.3.1 is always at path 2/3/1), requiring one metadata read then direct object access with no internal B-Tree traversal. This enables parallel reads: fetching 10 chunks is 10 independent GET requests that happen simultaneously.

> [!question]- Q: When is Zarr preferred over COG for satellite imagery?
> A: Zarr is preferred for large stacks of satellite imagery spanning many timesteps (e.g., all Landsat observations over a region from 1984 to present) pre-packed into a data cube aligned to a common grid. COG+STAC per scene is the standard for individual scenes; Zarr datacubes are faster to query but expensive to build and maintain.

> [!question]- Q: What Python library is Zarr almost always used with?
> A: Zarr is almost always used with Xarray, which uses Zarr as a storage backend the same way Pandas uses Parquet. Xarray provides the labeled dimension interface on top of Zarr's chunked storage.
