---
aliases:
  - Spatial Index
---


Geospatial indexes solve a fundamental problem: standard database indexes ([[B-Tree]]s) work on *one-dimensional* ordered values, but spatial data is inherently *two-dimensional.*
- You can't sort points by latitude and longitude simultaneously in a way that preserves proximity!

[[R-Tree]] is the dominant spatial indexing structure.
- The idea is to ==group nearby geometries into minimum bounding rectangles (MBRs), then group those MBRs into larger MBRs, recursively==.
- ==Queries work by traversing the tree==: If a query rectangle doesn't intersect an MBR, discard that whole subtree.
- R-Trees ==handles all geometry types==: points, lines, polygons, and support queries like intersection, containment, distance.
- Weakness: MBRs can overlap heavily in dense or irregular datasets, causing the tree to check many branches unnecessarily.

[[PostGIS]] uses a variant called [[R-Tree]] with [[GiST]]; GiST is a PostgreSQL framework for building custom index types, and PostGIS plugs spatial logic into it.

PostGIS actually offers two index types:
- [[GiST]]: Balanced tree, good for overlapping geometries, handles all shapes well.
	- For most cases, GiST is the right choice.
	- (Postgres implements an R-Tree on top of the GiST framework)
- [[GiST|SP-GiST]] (Space-partitioning GiST): Partitions space into non-overlapping regions, which is better for point data with clustering, worse for polygons.


[[QuadTree]]s recursively subdivide space into 4 quadrants.
- Simpler than [[R-Tree]]s, works well for uniformly distributed point data.
- Degrades badly with clustered data or irregular shapes: heavily populated areas keep subdividing while empty areas waste tree depth.
- Used internally in some systems, but rarely exposed directly as a database index type.

[[KD-Tree]]s are binary trees that alternate splitting on X and Y axes at each level. Great for nearest-neighbor queries on point data, less good for range queries and polygons. Common in in-memory spatial libraries (scipy, sklearn), but not typically used in databases.


## Two-Phase Query Execution
- Almost all spatial indexes work in two phases, which is important to understand:
	1. Phase 1 (index): Find candidate geometries whose MBR intersects query.
	2. Phase 2(filteR): Test the actual geometry against query predicate.
- The index gives you the candidates fast but imprecisely, since MBRs are approximations. The second phase does exact geometric computation on only those candidates.
- In PostGIS, the && operator uses the index, and ST_Intersects (e.g.) does the exact test. Running ST_Intersects *without* an index forces exact computation on every row.


## Practical PostGIS Notes
- Always create a spatial index on *geometry columns* that you'll query!
- Spatial indexing is still an active research area because no single structure handles all geometry types, distributions, and query patterns equally well, but PostGIS with GiST R-Trees are the safe default for most production use cases.


> Aside: Grid/Cell indexes ([[Discrete Global Grid System|DGGS]]-based) like [[Geohash]], [[H3]], [[A5]], [[S2 Geometry|S2]]
> - Encode geometries to cell IDs and typically use a regular [[B-Tree]] to index them.
> - Fast for point lookups and approximate proximity, and cheap to implement, but polygon queries require covering the polygon with cells, which get expensive for large or irregular shapes, and you get false positives at cell boundaries that require a second-pass filter. 
> - Not a replacement for "real" geospatial indices.








___________________________

GeoSpatial Indices have become a hot topic in System Design interviews because of proximity-based services like Yelp, Uber, and GoPuff, where we want to find "restaurants within 5 miles of me."

**TLDR: If you're asked about Geospatial Indexing in an interview, focus on explaining the problem clearly and contrasting a tree-based approach with a hash-based approach:**
> "Traditional indexes like ==B-trees== don't work well for spatial data because they treat latitude and longitude as ==independent dimensions==. To efficiently search for nearby locations, ==we need an index that understands spatial relationships==. ==Geohash== is a hash-based approach that ==converts 2D coordinates into a 1D string==, preserving proximity. This ==allows us to use a regular B-tree index on the geohash strings== for efficient proximity searches. However, ==tree-based approaches like R-trees can offer more flexibility and accuracy by grouping nearby objects into overlapping rectangles, creating a hierarchy of bounding boxes== (while also allowing for shapes)."

The naive approach would be to use a standard [[B-Tree]] index on latitude and longitude:
```sql
CREATE TABLE restaurants (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8)
);

CREATE INDEX idx_lat ON restaurants(latitude);
CREATE INDEX idx_lng ON restaurants(longitude);
```
- This falls apart quickly when we try to execute a proximity search -- think about how a B-Tree index on latitude and longitude actually works!
	- We're trying to solve a 2D spatial problem (finding points in a circle) using two separate 1D indexes!
	- We'll use the latitude indexes first, finding all the restaurants with the right latitude (a long strip spanning the entire globe), and then for each of these restaurants, we need to check if they're also in the right longitude range.
		- ==In fact, our index on longitude isn't helping, because we're not doing a range scan; we're doing point lookups on each restaurant we found in the latitude range!==
	- If we try to be clever and use both indexes together via an **index intersection**, the DB still has to merge two large sets of results, creating a rectangular search area much larger than our actual search cluster radius!
![[Pasted image 20250520182336.png]]
- This is why we need indexes that understand 2D spatial relationships; rather than treating latitude and longitude as independent dimensions, spatial indices let us organize points based on their actual proximity in space!


**THINK:** "Oh, but can't we use a [[Composite Index]] on (latitude, longitude) and query by that?" **Nope**! That's still not going to let us scan a contiguous, valid part of the index! 
If we had 
```
(-122.4194, 37.7749)  // San Francisco
(-122.4194, 37.7750)  // Slightly north in SF
(-122.4194, 37.7751)  // Even more north in SF
(-122.4193, 37.7749)  // Slightly east in SF
(-122.4000, 37.7749)  // Much further east
(-121.0000, 37.7749)  // Way across the bay
```
And then a query like:
```sql
-- Looking for points near (-122.4194, 37.7749) within ~0.01 degrees
SELECT * FROM restaurants
WHERE longitude BETWEEN -122.4294 AND -122.4094
AND latitude BETWEEN 37.7649 AND 37.7849;
```
- We still have to: Scan all longitude values in the range (-122.24294 to -122.4094), and then for each longitude value, check if the latitude is also in range.
	- We read the ROOT index page from disk into our. Since nodes in a B-Tree are stored on individual pages.
	- We read some INTERMEDIATE index page from disk into memory (assuming it's not in buffer )



Related Options:
- [[Geohash]]ing, [[QuadTree]], [[R-Tree]]