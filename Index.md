---
aliases:
  - Indexing
---
- [[Index]]es are data structures that are stored on disks and act as a map to tell us on which page some item exists in the database.

Indexes are stored on [[Page]]s, just like table data!
- e.g. For a B-Tree, nodes are typically the same size as dat pages (~4KB, 8KB, or 16KB, depending on the database)


Various types:
- [[B-Tree]]
- [[Log-Structured Merge Tree]] (+ [[Sorted String Table]])
- [[B+ Tree]]
- [[Hash Index]]
- [[Geohash]]+B-Tree
- [[QuadTree]]
- [[R-Tree]]
- [[Inverted Index]]
- [[Generalized Search Tree|GiST]]

#### Composite Indices
- A [[Composite Index]] is the most common optimization pattern that we'll encounter in practice.
- Instead of creating separate indexes for each column, we create a single index that combines multiple columns in a specific order.
- This matches how we typically query data in applications!
```sql
SELECT * FROM posts 
WHERE user_id = 123 
AND created_at > '2024-01-01'
ORDER BY created_at DESC;
```
If for this query we had two *separate* indices:
```sql
CREATE INDEX idx_user ON posts(user_id);
CREATE INDEX idx_time ON posts(created_at);
```
This wouldn't be optimal! The database would need to:
1. Use one index to find all posts by user 123
2. Use another index to find all posts after January 1
3. Intersect these results in-memory
4. Sort the final result set by created_at
**==Instead, a composite index would give us everything we need in one shot!==**
```sql
CREATE INDEX idx_user_time ON posts(user_id, created_at);
```
- ==When we create a composite index, we're really creating a B-Tree where each node's key is a concatenation of our indexed columns.==
- For our (user_id, created_at) index, each key in the B-tree is effectively a tuple of both values!
So the keys might conceptually look like:
```sql
(1, 2024-01-01)
(1, 2024-01-02)
(1, 2024-01-03)
(2, 2024-01-01)
(2, 2024-01-02)
(3, 2024-01-01)
```
Now when we execute our query, we can traverse the B tree to find the first entry for user_id=123, then scan sequentially through the index entries for that user until it finds entries beyond our date range. We get both our filtering and sorting for free!
- ==WARN:== The ==order of the columns in a composite index is crucial!==
	- Our index on (user_id, created_at) is great for queries that filter on usre_id filter, but it's not helpful for queries that only filter on created_at! This follows from how B-Trees work.
	- ==Tip:== "Order your columns in a composite index from most selective to least selective"


### Covering Indexes
- A [[Covering Index]] is one that includes all of the columns neeeded by your query -- not just the columns that you're filtering or sorting on!
- Think about a social media feed with post **timestamps** and **like counts**.
- With a regular index on (user_id, created_at), we can first find matching posts in the index, then we have to fetch each post's full data page just to get the like count -- that's a lot of extra disk reads just to display a number!
- ==By including the likes column directly in our index, we can skip these expensive page lookups entirely, returning everything we need straight from the index itself!==
```sql
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INT,
    title TEXT,
    content TEXT,
    likes INT,
    created_at TIMESTAMP
);

-- Regular index
CREATE INDEX idx_user_time ON posts(user_id, created_at);

-- Covering index includes likes column
CREATE INDEX idx_user_time_likes ON posts(user_id, created_at) INCLUDE (likes);
```
- Now Postgres can return results purely from the index data, with no need to look up each post in the main table! ==This is especially powerful for queries that only need a small subset of columns from large tables.==
- ==The tradeoff is size==: ==Covering indices are larger because they store extra columns==! For frequently-run queries that only need a few columns, the performance boost from avoiding table lookups often justifies the storage cost.
	- This is particularly true in social feeds, leaderboards, and other read-heavy features where query speed is critical.
- **WARN**: The reality in 2025 is that ==covering indexes are more of a niche optimization than a go-to solution==. Modern database query optimizers have become quite sophisticated at executing queries efficiently with regular indexes. ==While covering indexes can provide significant performance gains in specific scenarios== - like read-heavy tables with limited columns - ==they come with real costs== in terms of maintenance overhead and storage space.
	- In an interview, ==you may be wise to focus on simpler indexing strategies== and, if reaching for covering indexes, be sure to make sure you have a good reason for why it's necessary.
	- If you're not sure if they make sense in a given scenario, it's often better to err on the side of simplicity.


## Geospatial Indexes

| Index Type | PostgreSQL keyword      | Best for                                                               | Used in LA Observatory project for                      |
| ---------- | ----------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------- |
| B-Tree     | `USING BTREE` (default) | Equality, range, sort on orderable types                               | `h3_r*` columns, `created_at`, `sr_number`              |
| GiST       | `USING GIST`            | 2D geometry, PostGIS spatial queries                                   | `geom` column on `public.sr_311`                        |
| SP-GiST    | `USING SPGIST`          | Space-partitioning structures, H3 proximity                            | Not used (B-Tree sufficient for our H3 queries)         |
| GIN        | `USING GIN`             | Full-text search, JSONB containment, arrays                            | Not used in Phase 1                                     |
| BRIN       | `USING BRIN`            | Very large tables with natural ordering (e.g. time-series append-only) | Candidate for `_fetched_at` on raw tables               |
| Hash       | `USING HASH`            | Equality only, slightly faster than B-Tree                             | Rarely used — B-Tree does everything Hash does and more |


__________________

Let's understand the problem that indexing solves.

![[Pasted image 20260605181254.png]]
- Data in a database is often arranged in ~8KB [[Page]]s.
- To find a particular item in the database *without indexing,* we pull a page into RAM, into memory, we scan for the item, and if we don't find it, we put it back, pull out the next one, put it back, etc.
- If we had 100M users in our user table, and each page has 100 rows, that 1 Million pages! Each round trip from SSD to RAM is about 100ms, so that's 100 seconds in the worst case to find the item that we're looking for. This is far larger than the user wants to wait!

So how do indexes solve this problem for us?
- ==Indexes are just datastructres stored on disk that act as a map to tell us where items exist in the database.==
- When a query comes in for a particular item:
	- We first pull the index into memory
	- We check which page that index tells us the resource/item lives on
	- We pull only that particular page
- As opposed to reading up to 1,000,000 pages, we now use the index to tell us exactly which page to look at!

So what types of indexes are there, and when should we use them?

[[B-Tree]] is by far the most popular index; Basic tree structure where each node in the tree is a sorted list of values, with pointers to another page in disk; either a child node in the B-Tree, or an actual database [[Page]].
![[Pasted image 20260605181649.png]]
If we wanted to select all users with age=51, and in our `users` table, we built an index on the `age` column...
- We'd pull our root node (a [[Page]] itself) into memory, and we'd say: "We want 51, that's greater than 50, less than 90", and so we'd pull in that second-level B-Tree node (itself a [[Page]]) into memory, and look at it, and say, "well, 51 is less than 55", and so which page does that correspond to? It corresponds to a data Page 3!
- So we pull Page 3 into memory, which is where all of our users with this condition live.

What if we were to do a `WHERE age > 51`?
- We'd pull in our root page for our index
- We then pull in BOTH the (55, 67) and (91, 100, 102) pages into memory
- When then pull in all 7 data pages into memory.


The next index is a [[Hash Index]]
![[Pasted image 20260605182140.png]]
- A Hash Index is just a [[Hash Map]] on disk.
- If we're looking for a user, we pass their (e.g.) email into a hash function, and then we have some hashmap that maps this key to a value, where the value is just a pointer to where that data exists on disk.

==In reality... hash indexes are rarely used in production databases.==
- They offer O(1), but [[B-Tree]]s function nearly as well for exactly matches, and *also* support range queries!

The only place you'll see these are in-memory K:V stores like [[Redis]]



[[Geospatial Index]]es
![[Pasted image 20260605182326.png]]
- If you have Geospatial data, like trying to search within regions ... like getting the people in a certain raidsu
- Why can't we use B-Trees to satisfy this query?
	- ==B-Trees excel at one-dimensional data, but not two dimensional data==

This query would give us:
- All of the red data for latitude
- All of the blue data for longitude
And then we'd pull each of these strips into memory and do a merge. These strips are huge.

So is there a data structure that can help us?
Yes, [[Geospatial Index]]es and [[Discrete Global Grid System|DGGS]]s can help.

We'll look at:
1. [[Geohash]]ing
2. [[QuadTree]]s (not as popular as R-Trees)
3. [[R-Tree]]s


Geohashing
![[Pasted image 20260605192253.png]]
- Take map of the world, and split it into 4 parts
- Recursively split each of those cells
- By continuing to do this, we get increasing levels of precision
	- Maybe New Mexico is 31, and Albequerque specifically is 310
- Once you've converting lat/lon into these one-dimensional strings, ==all nearby locations share a similar prefix.==
	- ((This isn't totally true in all cases, as you can see with 300 vs 211... but this is going to be better for deeper layers))
- We can create the Geohashes and then build a [[B-Tree]] on top of the hashes themselves.

==NOTE:== For the illustration, we did it with simple numbers, but in reality, we  actually [[Base32]] encode them, so that Los Angeles is `9qh16`



Quadtrees
![[Pasted image 20260605192545.png]]
- Similar to Geohashing in that we split the world recursively
- We map the splitting to a tree, rather than a one-dimensional string.
	- We only need to go deeper in this tree in the places where we have high density. 
- Imagine that this was a mapping of the world, and each dot was a location in our database.
- We split the world into four initial grids, creating an initial tree whose root has four children.
	- These point to all the businesses in each respective cell
- For quadtrees we specify a `k` value: If any cell has >k items, recursively split it.
	- We can see how we split the initial blue cell, and then had to recursively split one of the child cells.
	- Now, when we want to find a particular business, we work our way down the tree accordingly...
	- This tree is the index that ends up being stored on disk

(==This is a shitty explanation. "The tree is stored on disk", okay how? How do we walk it? ==)
NOT USED IN PRODUCTION MUCH THESE DAYS; R-TREES ARE, INSTEAD.


[[R-Tree]]s

![[Pasted image 20260605193305.png]]
- Derived from Quadtrees, but instead of crudely splitting the world into even 4s, is more dynamic: Does some clustering to find locations or businesses that are close to eachother, and then each of these larger groupings can even have some overlap.
- It's the same general idea of working down a tree to get increasing precision... working down to finally get the locations.
- It's a bit more dynamic and fairly complex.

[[PostGIS]] uses [[R-Tree]]s via its own [[Generalized Search Tree|GiST]] index, an adaptable framework.



[[Inverted Index]]es
![[Pasted image 20260605193733.png]]
A [[B-Tree]] would be able to satisfy a query like `LIKE pizza%`, where we have a leading prefix that's known, since then it's just a range query on a collection of lexicographically sorted string...  But it can't do `%pizza%`

Instead, we can create an inverted index. Great anytime that you need to search for text.

We create a mapping from words/tokens to documents that they appear in.

