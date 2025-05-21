---
aliases:
  - Indexing
---
- [[Index]]es are data structures that are stored on disks and act as a map to tell us on which page some item exists in the database.



Various types:
- [[B-Tree]]
- [[LSM Tree]] (+ [[SSTable]])
- [[B+ Tree]]
- [[Hash Index]]
- [[Geohash]]+B-Tree
- [[QuadTree]]
- [[R-Tree]]
- [[Inverted Index]]

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