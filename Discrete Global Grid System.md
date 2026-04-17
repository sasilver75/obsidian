---
aliases:
  - DGGS
  - Grid System
---
A system that ==partitions== the entire Earth's surface into a ==hierarchy of discrete cells==, each with a ==unique identifier==.
- COMMON MISCONCEPTION: DGGSs don't replace [[Geospatial Index]]es. Replacing PostGIS with H3 cell IDs in a regular B-tree index doesn't give you full spatial query capability. You can do fast point lookups and approximate proximity queries, but complex operations like polygon intersections, buffer analysis, routing still need proper geometries. DGGSs are best thought of as ==complement== to spatial indexes, not a replacement.

Defining properties:
- Covers the entire globe without gaps or overlaps
- Hierarchical, with cells nesting within parents cells
- Each cell has a unique ID
- Cells at the same level are *roughly* equal area and shape

Major DGSS options:
- [[S2 Geometry]] (Cube and Hilbert), by Google: Square-ish cells on a cube projection, 64-bit integer IDs, great for indexing and region coverage; best for engineering/backend use.
- [[H3]] (Hexagons), by Uber: Hexagonal cells, which have uniform adjacency (all 6 neighbors equidistant, unlike squares), better for analytics, movement, and visualization, and widely used in mobility and geospatial analytics.
- [[A5]] (Pentagons), by the deck.gl guy: Newer DGGS that uses pentagons as base shape. tries to address H3's weakness, which is that H3 cells vary up to 2x in globally due to icosahedron-to-hex mapping; A5 aims for more consistent cell areas across the globe, but is still new/niche.
- [[Geohash]]: Older, simpler system using a Base-32 string; maps to rectangular cells on a flat projection. Widely supported but less accurate near poles, unequal areas. Still common in databases (Redis, Elasticsearch, MongoDB)
- DDGRID/ISEA (academic/standards-focused implementations; niche)



# ==Cell Indexing==/Geocoding to a cell grid and implications
- The process is usually called spatial indexing, geocoding to a grid cell, point-to-cell encoding, cell indexing, etc. It's the process of mapping a point to a grid cell.
	- The process somewhat informs the appropriate usage (along with other characteristics)
- In [[S2 Geometry|S2]]: 
	- S2's power of 4 subdivision means every level is a clean bit-prefix of the level below it.
	- We encode a point once at full resolution (level 30, which the [[Hilbert Curve]] passes through), and can derive any coarser cell by simply shifting bits, which is essentially free.
	- This makes S2 great for systems that need to work across multiple resolutions dynamically, or that need to answer: "Which level-N cell contains this point?" as a hot-path operation.
		- Google built S2 for this exact use case: Serving Maps queries at scale where you need fast, cheap spatial lookups across many zoom levels simultaneously.
		- The tradeoff is that S2 cells are square-ish and have somewhat unequal areas, which makes aggregation across large geographic extents slightly biased.
	- The resolution flexibility of S2 makes it ==the right choice for backend infrastructure like spatial databases, proximity indexes, and systems where you're constnatly moving between scales.==
- In [[H3]]:
	- Encodes directly to a target resolution, unlike S2. You specify the level up-front, and the algorithm runs to that level.
	- Moving between resolutions requires recomputation, not bit manipulation like in S2.
		- This initially sounds like a advantage, but in practice it rarely matters because H3's primary use case (for Uber) is **aggregation and analytics***, not real-time indexing: You pick a resolution appropriate for your analysis, encode everything at that resolutions, and stay there.
		- The hexagonal cell shape is the real story: every hexagon has exactly 6 neighbors all the same distance from the center, which squares and rectangles don't have. This makes H3 ideal for movement analysis, proximity calculations, and any problem where you're asking: "What's near this cell" (rideshare demand forecasting, disease spread modeling, delivery zone optimization). 
		- The ~2x area variation globally is a weakness for analyses that require strict equal-area comparisons across continents.
	- H3's fixed-resolution-first design reflects its ==analytics orientation==: You're choosing a meaningful geographic unit (roughly a neighborhood, a city block, a county) and aggregating data to it, not hopping between levels.
- In [[Geohash]]: 
	- The oldest of the above two, and the simplest. Encodes a point as a base-32 string where coarser resolutions are string prefixes (similar to S2); You can zoom out by truncation, just like in S2.
	- This makes it trivially easy to implement prefix queries in any database that supports string indexing, which is why historically Geohash has been the default choice for adding basic spatial capability to Redis, Elasticsearch, MongoDB, and similar systems.
	- Weaknesses are real though: Cells are rectangular on a Mercator projection, so areas vary badly near the poles, and the grid has a well-known edge discontinuity problem where two points right next to eachother can have a completely different geohash prefix if they straddle a cell boundary.
	- For serious spatial analysis, Geohash is showing its age, but for simple "find things near me" queries in a system that already has string indexing, it still works fine.
	- The pragmatic choice if you need spatial capability in a system that wasn't designed for it, and you don't want to add a dependency.





