  
  

# (1) SAM Click-to-Delieate 🥈

  A web app where you click anywhere on a satellite basemap and Meta's [[Segment Anything Model]] (SAM) runs inference on the underlying [[Cloud-Optimized GeoTIFF|COG]] tile, returning a delineated polygon — a field boundary, building footprint, or land cover patch — in under a second. Results export as [[GeoParquet]].

  Full stack:
  - Ingestion: [[SpatioTemporal Asset Catalog|STAC]] query against [[Element84 Earth Search|Element84]] or [[Microsoft Planetary Computer|Planetary Computer]] for [[Sentinel|Sentinel-2]] or [[National Agriculture Imagery Program|NAIP]] [[Cloud-Optimized GeoTIFF|COG]]s
  covering a target region
  - Backend: [[FastAPI]] serves tile requests to [[MapLibre GL JS]]; on click, extracts a pixel window from the COG via [[rasterio]] [[HTTP Range Request]], runs SAM inference, returns polygon [[GeoJSON]].
  - Storage: COGs on [[Amazon S3|S3]], inference results written to GeoParquet per session
  - Infrastructure: [[Docker]] + AWS [[Amazon Elastic Container Service|ECS]] or [[Amazon Lambda|Lambda]] for the inference endpoint, [[TiTiler]] for tile serving
  - Frontend: MapLibre GL JS — click on map → polygon animates onto the map within ~1 second, confidence heatmap as a second layer, [[GeoParquet]] download button.

  Why your background matters: Regrow was doing field delineation manually or with proprietary
  tools. This is the modern replacement for that workflow. You understand the domain problem —
  you're now building the infrastructure solution.

  Why Planet/Vantor cares: Field boundary delineation is one of the highest-value use cases for
  Planet's agricultural customers. Showing you can wire SAM into a COG-backed geospatial pipeline is directly relevant to their product work.

  CNG concepts: STAC discovery, COG HTTP range requests, ONNX inference on raster data, GeoParquet output, TiTiler.

  Frontend wow: Polygon appears on the map within a second of clicking. Nothing else in the
  geospatial portfolio world looks like this.

> Update from Opus: use `samgeo` specifically. SAM on raw satellite imagery is known to
  struggle because it was trained on natural imagery; samgeo is the community fork adapted for
  overhead. Also add a batch delineation mode — user drops a bounding box and it auto-delineates
  every field inside, outputting GeoParquet. Single-click is the demo; batch is the actual
  engineering.

  ---
# (2) Wildfire Burn Scar Monitor 🥉

  A pipeline that ingests [[Sentinel|Sentinel-2]] before/after scene pairs for recent California fires, runs a
  pre-trained burn severity classifier, stores results as a [[Zarr]] time series on [[Amazon S3|S3]], and serves an
  animated map showing fire progression with neighborhood-level impact statistics.

  Full stack:
  - Ingestion: [[Dagster]] pipeline — [[SpatioTemporal Asset Catalog|STAC]] query against [[Element84 Earth Search|Element84]] for Sentinel-2 L2A scenes bracketing a fire event, cloud masking, NBR computation, write monthly composites to Zarr on S3
  - ML inference: Pre-trained burn severity classifier (HuggingFace) run against the Zarr stack,
  outputting per-pixel severity scores.
  - Storage: Zarr for raster time series, [[GeoParquet]] for neighborhood-level aggregates ([[H3]] res-8), [[Cloud-Optimized GeoTIFF|COG]] for serving
  - Infrastructure: Dagster orchestration (connects to your Airflow experience), [[Docker]], S3, [[TiTiler]]
  - Frontend: [[MapLibre GL JS]] — animated time slider showing fire spread week by week, before/after swipe panel on raw imagery, neighborhood cards showing % burned area and NDVI recovery index

  Why your background matters: The Dagster pipeline here is structurally identical to the Airflow
  ETL work you did at Niantic — dependency-aware, parameterized by event/date, with downstream ML consumers. You already know how to think about this architecture.

  Why Planet/Vantor cares: Wildfire monitoring is an active product area. Planet sells daily imagery
   specifically for this use case. Showing you can build the full pipeline from STAC → Zarr →
  inference → serving is directly relevant.

  CNG concepts: Zarr + [[Xarray]] for raster time series, STAC temporal queries, COG serving, Dagster
  orchestration, H3 aggregation.

  Frontend wow: Animated fire spread over time, neighborhood-level impact cards, before/after swipe.

> Update from Opus: Wildfire Burn Scar Monitor → drop the "pre-trained burn severity" framing. Use real dNBR math + Copernicus EMS event list. Add the LLM angle: auto-generated damage reports per burned neighborhood using an LLM over the structured spatial data ("Glass Fire burned 1,203 residential structures across 47 km² in Napa County, with 89% of canopy in the affected area destroyed"). Suddenly your NLP background is visible.
  ---
# (3) Vessel Dark Activity Detector 🥈

  Ingest [[Automatic Identification System|AIS]] vessel position data, compute trajectory features using [[DuckDB]] window functions, score vessels with a pre-trained [[Isolation Forest]] anomaly model, and stream results to an animated [[deck.gl]] map with a live anomaly feed.

  Full stack:
  - Ingestion: AIS CSV from MarineCadastre → DuckDB, partitioned by date and vessel MMSI, written to [[GeoParquet]] on [[Amazon S3|S3]]
  - Feature engineering: DuckDB spatial window functions — speed variance, course change rate, time dark, port proximity, loitering score — all computed in SQL against the GeoParquet
  - ML inference: Pre-trained Isolation Forest (scikit-learn, serialized to S3), scored against
  feature vectors per vessel per day
  - Backend: [[FastAPI]] — historical track endpoint (DuckDB query against GeoParquet), [[Websockets|Websocket]]
  endpoint replaying recent vessel positions with live anomaly scores.
  - Infrastructure: [[Docker]], AWS [[Amazon Elastic Container Service|ECS]], S3, DuckDB as the query engine (no PostGIS server needed)
  - Frontend: deck.gl TripsLayer on [[MapLibre GL JS]] — animated vessel trails, anomalous vessels
  highlighted red, sidebar live feed of flagged vessels, click panel showing feature breakdown

  Why your background matters: The DuckDB pipeline here is a direct evolution of the ETL work at
  Niantic — but with spatial operations and no server needed. Your event-sourced architecture
  experience at Indigo maps directly onto the streaming anomaly feed pattern.

  Why Planet/Vantor cares: Maritime domain awareness is a major geospatial intelligence use case.
  Dark vessel detection specifically is used for sanctions enforcement, IUU fishing, and cargo
  insurance. The animated TripsLayer is one of the most impressive geospatial demos that exists.

  CNG concepts: GeoParquet at scale, DuckDB spatial as a serverless query engine, deck.gl
  high-density rendering, WebSocket streaming architecture.

  Frontend wow: Animated vessel trails across the ocean, live anomaly feed updating in real time,
  click-to-explain feature breakdown.

> Update from Opus:   Vessel Anomaly Detector → be honest it's a fit, not pre-trained. Or switch to a CLIP-style approach: embed trajectories, anomalies are the ones that don't cluster with normal shipping lanes. Vector search against a known-normal corpus. This leverages your IR background honestly.

  ---
# (4) Urban Canopy Equity Dashboard 🥉

  Stream [[3DEP]] [[Light Detection and Ranging|LiDAR]] [[Cloud-Optimized Point Cloud|COPC]] tiles directly from [[Amazon S3|S3]] using [[Point Data Abstraction Library|PDAL]] without downloading them, generate a canopy height model, aggregate to [[H3]] hexes, and visualize tree cover inequality overlaid with census income data and heat risk.

  Full stack:
  - Ingestion: PDAL pipeline streaming COPC tiles from USGS S3 bucket via [[HTTP Range Request]]s — no full download, spatial filter to target neighborhood boundaries
  - Processing: Ground/vegetation return classification, 1m CHM raster generation via [[rasterio]], H3
  res-9 aggregation of mean/max canopy height
  - Data join: Census API for median household income per block group, spatial join to H3 hexes via [[Geopandas]], written to GeoParquet
  - Storage: COG for the raw CHM on S3, GeoParquet for H3 aggregates, [[PMTiles]] via [[Tippecanoe]] for the vector layer
  - Infrastructure: Docker, S3, [[TiTiler]] for COG serving, FastAPI for H3 query endpoint
  - Frontend: [[MapLibre GL JS]] with 3D extruded H3 hexes colored by canopy height, income overlay toggle, neighborhood comparison panel — scroll across the city and watch canopy collapse as you enter lower-income areas

  Why your background matters: The PDAL pipeline is structurally similar to the data integration
  work at Regrow — reading from external data sources, transforming, and writing to a standardized format for downstream modeling. The equity framing shows you think about data in context, not just technically.

  Why Planet/Vantor cares: Urban tree canopy is an active product area for climate adaptation and
  environmental justice customers. COPC streaming is a core skill for anyone working with Planet's
  LiDAR products or 3DEP-derived data.

  CNG concepts: COPC streaming without full download, PDAL pipelines, [[Canopy Height Model|CHM]] generation, PMTiles for serverless vector tiles, H3 aggregation.

  Frontend wow: 3D extruded hexes, income/canopy correlation visible as you pan across the city.

  ---
# (5) Overture + DuckDB Neighborhood X-Ray 🥈

  Query [[Overture Maps Foundation|Overture Maps]] [[GeoParquet]] directly from [[Amazon S3|S3]] using [[DuckDB]] — no ETL, no local download — to compute neighborhood character scores (POI mix, building density, road connectivity) aggregated to [[H3]] hexes, served as [[PMTiles]] with a [[FastAPI]] query backend.

  Full stack:
  - Compute: DuckDB spatial extension queries Overture GeoParquet on S3 directly — places,
  buildings, transportation themes — no download step
  - Transformation: [[dbt]] models on DuckDB compute H3 res-9 aggregates — POI category entropy, building density, road network density, walkability proxy
  - Storage: Materialized GeoParquet on local/S3, PMTiles via [[Tippecanoe]] for the H3 hex layer
  - Backend: FastAPI endpoint — given a bounding box, DuckDB queries GeoParquet and returns
  neighborhood breakdown JSON in ~1 second
  - Infrastructure: [[Docker]], S3, no tile server needed (PMTiles served directly from S3 via
  Cloudflare R2)
  - Frontend: [[MapLibre GL JS]] — type any city, H3 hexes populate showing neighborhood character, query timing displayed ("2.1s, 847M records scanned"), click hex for POI breakdown donut chart

  Why your background matters: The dbt transformation layer maps directly onto your data engineering experience. The "no ETL, query in place" pattern is the architectural evolution of the painful data handoffs you eliminated at Regrow.

  Why Planet/Vantor cares: DuckDB + GeoParquet + PMTiles is the modern serverless vector analytics stack. This project demonstrates you understand why the architecture is designed the way it is, not just how to use the tools.

  CNG concepts: DuckDB querying remote GeoParquet, dbt for documented transformations, PMTiles as serverless vector tiles, Overture Maps data model.

  Frontend wow: Real-time query timing display, POI breakdown charts, city-scale H3 visualization.

  ---
# (6) InSAR Subsidence Time Machine 🥉

  Process [[Sentinel|Sentinel-1]] [[InSAR]] displacement products for a subsiding city (Las Vegas, Central Valley,
  Jakarta) into a [[Zarr]] time series on [[Amazon S3|S3]], detect anomalous sinking zones, and build a map where
  clicking any point shows millimeter-precision ground deformation over years.

  Full stack:
  - Ingestion: ASF HyP3 API (free, 27GB/month) submits Sentinel-1 InSAR jobs, downloads [[GeoTIFF]]
  displacement products, converts to COGs on S3
  - Processing: [[Xarray]] + [[Zarr]] — stack displacement rasters into a time series cube on S3, compute per-pixel cumulative displacement and velocity
  - ML inference: Pre-trained change point detection (ruptures library) identifies anomalous
  acceleration in subsidence time series
  - Storage: Zarr on S3 for the time series cube, GeoParquet for anomaly zones, COG for velocity
  magnitude raster
  - Infrastructure: [[Docker]], S3, [[TiTiler]] for [[Cloud-Optimized GeoTIFF|COG]] serving, FastAPI for time series endpoint
  - Frontend: MapLibre GL JS — city map colored by subsidence velocity, click any point → FastAPI reads Zarr time series → animated chart shows mm-level deformation since 2017, anomaly zones pulse

  Why your background matters: The Zarr time series architecture is structurally identical to the
  simulation output pipelines at Regrow — time-indexed, multi-variable, needs efficient partial
  reads. You've solved this problem in a different domain.

  Why Planet/Vantor cares: Ground deformation monitoring is directly relevant to infrastructure risk
   assessment, insurance underwriting, and climate adaptation — all active Planet customer segments.

  CNG concepts: Zarr + Xarray for raster time series, STAC for SAR product discovery, COG serving, change point detection on array data.

  Frontend wow: Click anywhere on a city → watch a chart animate showing how many mm that spot has sunk since 2017.

  ---
# (7) LA Shade & Cool Route Finder 🥉

  Given start/end addresses and a time of day, compute a walking route that maximizes shade using building shadows from [[Light Detection and Ranging|LiDAR]] heights plus tree canopy, with sun position from [[Ephemeris]]. Render the route colored by per-segment shade score with a feels-like temperature estimate.

  Full stack:
  - Data prep: USGS [[3DEP]] [[Cloud-Optimized Point Cloud|COPC]] via [[Point Data Abstraction Library|PDAL]]→ building heights + tree canopy → precomputed shadow rasters at 15-minute intervals stored as COGs on S3
  - Routing: OSM road network loaded into [[PostGIS]] + [[pgRouting]], edges weighted by shade score at requested time.
  - Backend: FastAPI — given (start, end, time), queries [[PostGIS]] for shaded route, returns GeoJSON with per-segment shade scores
  - Infrastructure: [[Docker]], PostGIS on [[AWS Relational Data Store|RDS]] or local, [[Amazon S3|S3]] for [[Cloud-Optimized GeoTIFF|COG]]s, [[TiTiler]]
  - Frontend: [[MapLibre GL JS]] — route drawn colored green (shaded) to red (exposed), time-of-day slider updates shading live as shadows shift, feels-like temperature per segment

  Why your background matters: The routing + real-time query pattern maps onto the logistics
  platform work at Indigo. The precomputation vs. on-the-fly tradeoff is a genuine system design
  problem you've navigated before.

  Why Planet/Vantor cares: Urban heat and walkability are active product areas. This project shows
  LiDAR processing + routing + real-time serving in a single pipeline — rare combination.

  CNG concepts: COPC streaming, solar geometry computation, COG precomputation strategy, pgRouting, PostGIS.

  Frontend wow: Route recolors in real time as you drag the time slider — shadows visibly shift.

  ---
# (8) Agricultural Field Change Detector 🥉

  A pipeline that monitors a set of agricultural fields over a growing season using [[Sentinel|Sentinel-2]] time
  series, detects anomalous spectral changes (crop stress, flooding, harvest timing), and serves a
  farmer-facing dashboard showing field health history.

  Full stack:
  - Ingestion: [[SpatioTemporal Asset Catalog|STAC]] query against [[Element84 Earth Search|Element84]] for Sentinel-2 scenes intersecting field boundaries
  ([[GeoParquet]]), cloud mask, compute [[Normalized Difference Vegetation Index|NDVI]]/[[Normalized Difference Water Index|NDWI]]/NDRE per field per scene, write to Zarr on S3
  - ML inference: Pre-trained crop stress classifier (ONNX from HuggingFace) run against
  spectral-temporal feature vectors per field
  - Orchestration: [[Dagster]] pipeline — parameterized by growing season, field set, and satellite —
  mirrors the Airflow work at Niantic but geospatially aware
  - Storage: [[Zarr]] for the time series cube, GeoParquet for field-level results, PMTiles for the
  field boundary layer
  - Backend: FastAPI — field health history endpoint, anomaly alerts endpoint
  - Frontend: [[MapLibre GL JS]] — field polygons colored by current health score, click a field → time series chart of NDVI through the season, anomaly flags shown as markers

  Why your background matters: This is literally what Regrow was doing. You understand the domain, the customer (farmers/agronomists), and the pain of slow simulation turnaround. You're now building the cloud-native version of that pipeline.

  Why Planet/Vantor cares: Agriculture is Planet's largest commercial vertical. This project
  demonstrates you can build an end-to-end EO analytics pipeline for their core customer segment.
  The Zarr + STAC + Dagster stack is exactly what their engineering teams use.

  CNG concepts: STAC temporal queries, Zarr time series, Dagster orchestration, ONNX inference on raster data, PMTiles for field boundaries.

  Frontend wow: Field polygons pulse based on current health. Click a field → animated NDVI chart
  through the growing season.

> Update from Opus: this one I'd actually cut in favor of one of the new ones
  below. It's too narrow-domain for non-ag companies and the ML story is weak without training

---
# (9) Planet Scene Browser 🥉

  A full [[SpatioTemporal Asset Catalog|STAC]] browser UI that queries multiple catalogs simultaneously (Planet STAC API, [[Element84 Earth Search|Element84]], [[Microsoft Planetary Computer|Planetary Computer]]), lets you filter by cloud cover, date, sensor type, and draw an [[Area of Interest|AOI]], then previews [[Cloud-Optimized GeoTIFF|COG]] thumbnails and downloads scenes directly to [[Amazon S3|S3]].

  Full stack:
  - Backend: [[FastAPI]] wrapping multiple STAC clients (pystac-client) — unified search across
  catalogs, signed URL generation for COG access, scene download orchestration to S3
  - Processing: On download, rasterio validates and converts to COG if needed, registers in a local STAC catalog (stac-fastapi)
  - Storage: S3 for downloaded scenes, local PostgreSQL-backed STAC catalog
  - Infrastructure: [[Docker]], S3, stac-fastapi, [[TiTiler]] pointed at downloaded COGs
  - Frontend: [[MapLibre GL JS]] — draw AOI on map, date/cloud cover sliders, results grid showing COG thumbnails, click to preview full scene via TiTiler, download to S3 button

  Why your background matters: This is a platform tool — exactly the kind of developer
  infrastructure you built at Niantic. The multi-source STAC abstraction layer maps onto your
  experience building unified APIs over disparate backend systems.

  Why Planet/Vantor cares: You'd be building a simplified version of Planet's own Explorer product.
  Understanding STAC deeply enough to build a browser over it signals you understand their core data infrastructure.

  CNG concepts: STAC spec deeply, pystac-client, COG thumbnail extraction, stac-fastapi, TiTiler,
  multi-catalog federation.

  Frontend wow: Draw a box on a map → grid of satellite imagery thumbnails appears instantly. Click one → full resolution preview loads via COG range request.

____________
(Below ones generated with 4.7 Opus, above )
# (10) Natural Language Earth Observation ("Chat with EO") 🥇

  A chat interface that translates natural language questions into STAC queries, spatial filters,
  and temporal constraints. "Show me all Sentinel-2 scenes over Central Valley agricultural areas in July with less than 10% cloud cover" → the system issues STAC queries, pulls COGs, computes
  indices, renders a map.

  Full stack:
  - NLP layer: DSPy module — parses query → structured plan (collections, bbox, time range, filters, computation). Uses few-shot examples. Optimized with DSPy's optimizers for routing accuracy.
  - Spatial resolution: Geocoding pipeline — gazetteer lookup for place names, OSM for admin
  boundaries, fallback to LLM for fuzzy descriptions ("the Central Valley")
  - Execution: pystac-client executes the plan, DuckDB filters results, rasterio/xarray computes
  indices, outputs COG + GeoParquet
  - Backend: FastAPI + DSPy + LangSmith for tracing
  - Frontend: Chat UI + live map + a "reasoning trace" panel that shows each step — what the LLM inferred, what queries it ran, what it found

  Why this is the strongest project for you:
  - Directly leverages DSPy, RAG/IR, NLP (your MS thesis territory)
  - Genuinely differentiated — Planet doesn't have this yet, Element84 doesn't, neither does
  Descartes
  - CNG purity — every step is STAC + COG + GeoParquet
  - The reasoning trace panel is an unusual "wow" — you're showing the LLM's work

  Frontend wow: Type a vague question → map populates in real time with a sidebar explaining what the system did.

______________

# (11) Satellite Image Semantic Search 🥇

  Index a corpus of satellite chips using CLIP (or a satellite-specific variant like RemoteCLIP or
  SkyCLIP) embeddings. Allow text queries: "large solar farms", "flooded agricultural fields",
  "clearcut patches near rivers". Results render on a map with similarity scores.

  Full stack:
  - Ingestion: Chip a region of NAIP or Sentinel-2 into tiles, embed each tile with RemoteCLIP
  (pre-trained, real, checkpoints on HuggingFace)
  - Vector store: pgvector or Qdrant, with H3 index for spatial lookup
  - Query: Text → CLIP text embedding → vector search → ranked chip locations
  - Backend: FastAPI, pgvector, TiTiler
  - Frontend: Search box + map with top-K results highlighted, click a result → COG tile preview

  Why this is great for you:
  - Uses IR fundamentals (embeddings, ANN search) in a novel domain
  - RemoteCLIP is real and pre-trained — no ML training
  - Vector search + spatial filtering is a genuine engineering challenge
  - Impressive demo: "find me brick kilns in Bangladesh" actually works

  Frontend wow: Type a weird query → map populates with overhead thumbnails that match. It feels like magic.


________

# (12) Spatial RAG for Geospatial Developer Tools 🥈

  A RAG system over CNG ecosystem documentation — STAC spec, Overture schema, GDAL docs, rasterio, PDAL, FlatGeobuf spec, CNG Foundation docs, GeoPandas, dbt-duckdb, etc. Developer asks "how do I read a COG from S3 with a specific window in rasterio" and gets a grounded answer with citations and runnable code.

  Full stack:
  - Ingestion: Crawl ~20 key geospatial docs sites, chunk, embed with a code-aware embedding model, store in pgvector
  - RAG pipeline: DSPy — query understanding, retrieval, reranking, answer generation with citations
  - Eval: Small eval set of real developer questions, measure retrieval quality + answer correctness
  - Frontend: Chat UI with citations, code block extraction, "try this" button that runs the code
  against a sandbox

  Why this is great for you:
  - You have IR, RAG, DSPy, Cohere Scholar — this is your home turf
  - Useful to you personally while you're learning
  - CNG Foundation mentorship context: a tool that helps people learn CNG is the highest-signal
  project for that specific application
  - Has an eval story (DSPy optimization) that shows rigor

  Frontend wow: This has the least pure-geospatial wow but the most compelling
  mentorship-application wow — "I built a tool to help people learn this ecosystem."

_______________

# (13) OSM Real-Time Change Feed Visualizer 🥈

  Ingest OpenStreetMap's minutely replication feed, compute diffs, classify each change by type (new building, road edit, tag change), detect suspicious editing patterns (mass edits, coordinated
  vandalism), and render as an animated live map.

  Full stack:
  - Streaming: Kafka consumer against OSM minutely diff feed (real, openstreetmap.org publishes this)
  - Processing: Parse osmChange XML, spatial enrichment via H3, classification logic, write to GeoParquet + publish to WebSocket
  - ML inference: LLM-based classification of commit messages + edit patterns to detect vandalism signals (this is a real OSM operational problem)
  - Storage: GeoParquet partitioned by hour + H3, DuckDB for ad-hoc queries
  - Frontend: MapLibre + deck.gl — edits flash onto the map as they happen worldwide, filter by edit type, sidebar of suspicious patterns

  Why this is great:
  - Kafka/streaming story is genuinely differentiated (most geospatial portfolios are batch)
  - Real-time globe with edits flashing is a dramatically different "wow" from static maps
  - OSM quality is a live problem — Overture, HERE, Mapbox all care about this
  - Uses Kafka from your skills list honestly

  Frontend wow: Animated globe with edits pinging in real time — visually unlike any other portfolio
   project.


___________

# (14) Planet Constellation Tracker (3D) 🥇

  A real-time 3D globe showing the positions of Planet's ~200-satellite Dove constellation plus     
  SkySats, with orbit lines, ground tracks, and the ability to click a satellite for details — or   
  click a point on Earth to see which satellite is imaging it right now.                            


  Full stack:                                                                                       
  - Backend: Python + Skyfield (or poliastro) for orbital propagation. Ingests TLEs from Celestrak  
  for all commercial EO constellations (Planet, Maxar, Capella, ICEYE, BlackSky). Pre-computes      
  positions at a fixed cadence, exports as CZML.                                                    
  - Data bridge: Optional — hook into Planet's STAC API (planetarycomputer.microsoft.com or Planet's own) to show actual recent captures as footprints on the globe, linked to the satellite that took them                                                                                             
  - Scheduling math: Compute next pass over a user-drawn AOI across the constellation ("Dove-0c45 passes over LA in 47 minutes at 5.3° elevation")                                                  
  - Infrastructure: Dagster or cron to refresh TLEs every ~6 hours, S3 to host CZML, Cloudflare CDN 
  - Frontend: CesiumJS — spinning globe with ~400 satellites orbiting live, orbit trails, ground   
  tracks, click-to-detail panel, AOI-drawing tool that shows upcoming passes in a timeline.                    
  Why this is maximally on-target for Planet:                                                       
  - It is literally a visualization of their business                                               
  - Shows you understand orbital mechanics + constellation operations                               
  - Works equally well as a Vantor pitch (constellation intelligence is their whole vertical)
  - The 3D globe with satellites whipping around is one of the most viscerally impressive demos you can build.
  
  Frontend wow: Full-screen spinning Earth with hundreds of satellites in orbit, click one and the  
  camera smoothly follows it. Draw a polygon and a timeline appears showing the next 12 hours of   
  passes.                                                                                         
  
  CNG concepts: STAC catalog integration, time-dynamic data formats (CZML), constellation design fundamentals, footprint geometry.                                                        

______________________

# (15) 3D Damage Assessment Over Photorealistic 3D Tiles 🥈

  Take a real disaster event (Palisades Fire, 2024 Maui fires, Helene flooding) and fly through the
  impact zone in photorealistic 3D — buildings from Google's Photorealistic 3D Tiles, with          
  per-building damage scores overlaid from satellite-derived change detection.

  Full stack:                                                                                     
  - Ingestion: STAC query for Sentinel-2 + Planet (if available) before/after pairs, compute dNBR or change index                                                                                     
  - Segmentation: samgeo on post-event imagery against Microsoft/Google building footprints to
  extract per-building damage signal                                                                
  - Storage: GeoParquet of scored buildings, COGs of the change raster                              
  - Backend: FastAPI for building-level query endpoint                
  - Frontend: CesiumJS with Google Photorealistic 3D Tiles as base, GeoParquet building polygons extruded and colored by damage severity, time slider to flip between pre- and post-event imagery draped on terrain                                                                                 

  Why this is impressive:                                                                           
  - Combines cloud-native pipeline (STAC → COG → GeoParquet) with the most photorealistic geospatial visualization available                                                              
  - Flying through an actual damaged neighborhood in 3D is emotionally visceral in a way flat maps aren't.                                                                            
  - Photorealistic 3D Tiles + overlay data is exactly the pattern digital-twin companies are chasing

  Frontend wow: Camera swoops over a fire-damaged neighborhood with real 3D buildings, each tinted red-to-green by a damage score.                                                                   

_________



