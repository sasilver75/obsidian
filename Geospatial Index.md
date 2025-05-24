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