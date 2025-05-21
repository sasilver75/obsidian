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


Related Options:
- [[Geohash]]ing, [[QuadTree]], [[R-Tree]]