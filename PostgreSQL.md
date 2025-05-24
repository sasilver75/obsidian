---
aliases:
  - Postgres
---
SDIAH: https://www.hellointerview.com/learn/system-design/deep-dives/postgres


While Postgres is packed with features and is perhaps the most beloved database, your interviewer isn't looking for a database administrator; they want to see that you can make informed architectural decisions.
When should you choose to use PostgreSQL? When should you look elsewhere? What are the key tradeoffs to consider?

**==WARN==**: Candidates often get tripped up here, ==diving too deep into PostgreSQL internals== (e.g. talking about [[Multiversion Concurrency Control]] (MVCC) or the [[Write-Ahead Log]]), when your interviewer just wants to know if it can handle their data relationships, or ==making overly broad statements== like "NoSQL scales better than PostgreSQL" without understanding the nuances.


# Motivating Example
- Imagine we're designing a **social media platform**, not a massive one like Facebook, but one that's growing and needs a solid foundation.
- Our platform needs to handle some fundamental relationships:
	- **Users** can create **posts**
	- **Users** can **comment** on **posts**
	- **Users** can **follow** other **users**
	- **Users** can **like** both **posts** and **comments**
	- **Users** can create **direct messages** (DMs) with other **users**

What makes this interesting from a database perspective?
==**Different operations have different requirements!**==
- Multi-step operations like creating DM threads need to be **==atomic==** (creating a thread, adding participants, and storing the first message **must** happen together!)
- Comment and follow relationships need **referential integrity** (you can't have a comment without a valid post, or follow a non-existent user)
- Like counts can be **[[Eventual Consistency|Eventually Consistent]]** (it's not critical if it takes a few seconds to update)
- When someone requests a user profile, we need to **efficiently fetch** their recent post, follow counts, and other metadata
- Users need to be able to **search** through posts and find other users
- As our platform grows, we'll need to handle more data and more complex queries.

==This combination of requirements (complex relationships, mixed consistency needs, search capabilities, and room for growth) is a perfect example for exploring Postgres's limitations and strengths!==


# Core Capabilities and Limitations

## (1/4) Read Performance
- ==For most applications, reads vastly outnumber writes.== In our social media examples, users browse more than they post.
- When a user fetches a profile, we need to efficiently fetch all posts by that user
	- Without proper indexing, PostgreSQL would need to scan every row in the posts table to find matching posts. This is terrible!
	- We need [[Index]]es!
		- By creating an index on the user_id column of our posts table, we can quickly locate all posts for a given user without scanning the entire table.

##### Basic Indexing
- By default, Postgres uses [[B-Tree]] indexes, which work great for:
	- ==**Exact match queries**==: (WHERE email = ...)
	- ==**Range queries**==: (WHERE created_at > ...)
	- ==**Sorting**==: (ORDER BY username, assuming that username has a usable index on it)

By default, PostgreSQL will create a B-Tree index on your [[Primary Key]] column, but you also have the ability to create **Secondary Indexes** on the other column as well.
```sql
-- This is your bread and butter index
CREATE INDEX idx_users_email ON users(email);

-- Multi-column indexes for common query patterns. Here, we might want want to get all of the users posts, ordered by recency! Note that this also works for queries that JUST need user_id, but NOT those that JUST need created_at
CREATE INDEX idx_posts_user_date ON posts(user_id, created_at);
```
**==WARN:==** A common mistake is to suggest adding indexes for every column.
- Indexes **make writes slower** (as the index must be updated)
- **Takes up disk space**
- **May not even be used** if the query planner thinks a sequential scan would be faster

##### Beyond Basic Indexes
- Postgres shines for its support for specialized indexes!
	- These come up frequently in SD interviews because they can eliminate the need for separate specialized databases.
- [[Full-Text Search Index|Full-Text Search]] using [[GIN Index]]es!
	- Postgres supports FTS out of the box using Generalized Inverted Index (GIN) indexes, which work like the index at the back of a book.
	- This is perfect for when you need to find documents containing specific words.
```sql
-- Add a tsvector column for search
ALTER TABLE posts ADD COLUMN search_vector tsvector;
CREATE INDEX idx_posts_search ON posts USING GIN(search_vector);

-- Now you can do full-text search
SELECT * FROM posts 
WHERE search_vector @@ to_tsquery('postgresql & database');        
```
- For many applications, this means that you don't need a separate [[ElasticSearch]] cluster!
	- ==Use ElasticSearch when you need more sophisticated relevancy sorting, faceted search capabilities, fuzzy matching and "search as you type" features, real-time index updates, distributed search across very large datasets, and advanced analytics and aggregations.==
	- Postgres is good as a starter for when you have simpler search use cases.
- Postgres supports:
	- Word Stemming (finding/find/finds all match)
	- Relevance ranking
	- Multiple languages
	- Complex queries with AND/OR/NOT
- **==[[JSONB]]==** ==columns with GIN indexes== are particularly useful when you need flexible metadta on your posts.
	- In our social media example ,each post might have different attributes like
		- location
		- mentioned users
		- hashtags
		- attached media
	- Rather than give a separate column for each possibility, we can store this in a JSONB column, giving us the flexibility to add new attributes as needed, just like we would in a NoSQL database!
```sql
-- Add a JSONB column for post metadata
ALTER TABLE posts ADD COLUMN metadata JSONB;
CREATE INDEX idx_posts_metadata ON posts USING GIN(metadata);

-- Now we can efficiently query posts with specific metadata
SELECT * FROM posts 
WHERE metadata @> '{"type": "video"}' 
  AND metadata @> '{"hashtags": ["coding"]}';

-- Or find all posts that mention a specific user
SELECT * FROM posts 
WHERE metadata @> '{"mentions": ["user123"]}';
```
- ==Geospatial Search with [[PostGIS]]== extension is also a great feature!
	- PostGIS lets us index location data for efficient geospatial queries!
	- This is perfect for our social media platform, when we want to show users posts from their local area:
```sql
-- Enable PostGIS
CREATE EXTENSION postgis;

-- Add a location column to posts
ALTER TABLE posts 
ADD COLUMN location geometry(Point);

-- Create a spatial index
CREATE INDEX idx_posts_location 
ON posts USING GIST(location);

-- Find all posts within 5km of a user
SELECT * FROM posts 
WHERE ST_DWithin(
    location::geography,
    ST_MakePoint(-122.4194, 37.7749)::geography, -- SF coordinates
    5000  -- 5km in meters
);
```
- POSTGis is INCREDIBLY POWERFUL and can handle:
	- ==Different types of spatial data (points, lines, polygons)==
	- Various distance calculations (as the crow flies, ==driving distance==)
		- Uber initially used it for their entire ride-matching system, fun fact!
	- Spatial operations (==intersections==, ==containment==)
	- Different coordinate systems
- The index type used here [[GiST]] (Generalized Search Tree) for geometric data, using [[R-Tree]] indexing under the hood.



We can combine all these capabilities to create rich search experiences.
==**For example, we can find all video posts within 5km of SF that mention "food" in their content and are tagged with 'restaurant":**==
```sql
SELECT * FROM posts 
WHERE search_vector @@ to_tsquery('food') %% @@ is the full-text search match operator %% 
  AND metadata @> '{"type": "video", "hashtags": ["restaurant"]}' %% @> checks if the left JSONB contains the right JSONB %%
  AND ST_DWithin( %% ST_DWithin checks that two geography points are in a certain distance %%
    location::geography, %% ::geography casts our geography column to a geography type %%
    ST_MakePoint(-122.4194, 37.7749)::geography, %% Creates a point with longitude/latitude of San Francisco %%
    5000 %% Distance of 5000 meters %%
  );
```

##### Query Optimization Essentials
- So we've covered the different types of indexes that Postgres offered... but there's more to query optimization than just picking the right index type!
- [[Covering Index]]
	- When Postgres uses a n index to find a row, it typically needs to do two things:
		1. Look up the value in the index to find the row's location.
		2. Fetch the actual row from the table to get other columns you need.
	- **==But what if we could just store all the data we need right into the index itself?==**
		- This is what a CoveringIndex does!
			- ((I would imagine that this would be even faster than a HashIndex + Disk I/O))
	- Covering indexes can make queries significantly faster, because PostgreSQL can satisfy the entire query from just the index without touching the table. **==The tradeoff is that the index takes up more space, and that writing is a little slower.==**
```sql
-- Let's say this is a common query in our social media app:
SELECT title, created_at 
FROM posts 
WHERE user_id = 123 
ORDER BY created_at DESC;

-- A covering index that includes all needed columns
CREATE INDEX idx_posts_user_include 
ON posts(user_id) INCLUDE (title, created_at);
```
Above: **==See that the covering index can just have a subset of columns!==**

##### Partial Indexes
- Sometimes you only need to index a subset of data, for instance, if on our social media platform, **most queries are probably looking for active users, not deleted ones!**
```sql
-- Standard index indexes everything
CREATE INDEX idx_users_email ON users(email);  -- Indexes ALL users

-- Partial index only indexes active users
CREATE INDEX idx_active_users 
ON users(email) WHERE status = 'active';  -- Smaller, faster index
```
Above:
- See that we're creating a secondary index on our users table's email column, but it's a conditional index that's only used when the query has a WHERE status='active'! A [[Partial Index]] can be a great way to save space (and time).

##### Practical Performance Limits
- There's a good chance that during your non-functional requirements, you outlined some latency goals.
- These should get you started as some general rules of thumb:
	- **==Query Performance==**:
		- Simple indexed lookups: ==tens of thousands per second per core==.
		- Complex joins: ==thousands per second==
		- Full-table scans: Depends on whether the data fits in memory
	- **==Scale Limits==**:
		- Tables start getting unwieldy past ==100M rows==
		- Full-text search works well up to ==10Ms of documents==
		- Complex joins become challenging with tables >10M rows
		- Performance drops significantly when working set exceeds the available RAM.
==Memory is king when it comes to performance! Queries that can be satisfied from memory are orders of magnitude faster than those requiring disk access.==

## (2/4) Write Performance
- While reads might dominate most workloads, write performance is often more critical because it directly impacts user experience when it's not fast enough!
- **==When a write occurs in Postgres==**, a series of steps occur to ensure both performance and durability:
	- **==Transaction Log ([[Write-Ahead Log|WAL]]) Write (DISK):==** Changes are first written to the WAL on disk; this is a sequential write operation, making it relatively fast. The WAL is critical for durability. Once changes are written here, the transaction is considered durable because even if the server crashes, PostgreSQL can recover the changes from the WAL.
	- **==Buffer Cache Update (MEMORY):==** Changes are made to the data **pages** in PostgreSQL's shared [[Buffer Cache]], where the actual tables and indexes live in memory. When these pages are modified, they're ==marked as "dirty"== to indicate that they need to be written to disk eventually.
	- **==Background Writer (MEMORY -> DISK):==** Dirty pages in memory are periodically written to the actual data files on disk; asynchronously done through the background writer when memory pressure gets too high or when a checkpoint occurs. Batching these changes together gives better performance.
	- **==Index Updates (MEMORY AND DISK):==** Each index needs to be updated to reflect the changes. ==Like table data, index changes also go through the WAL for durability==. This is why having many indexes can significantly slow down writes -- each index requires additional WAL entries and memory updates!
		- (There's only one WAL in postgres; all changes, whether to tables or to indexes, are recorded in the same WAL.)
		- (All changes for a transaction (table and indexes) are written to the WAL as a single, atomic operation, and then they're later all flushed to disk together. The commit is only acked to the client after all relevant WAL records are safely on disk, along with a commit record to the WAL. On crash recovery, we replay the WAL, and only transactions with a valid, fully-written commit record in the WAL are considered committed and are applied. If a transactions commit record is missing, all its changes are ignored/rolled back.)

**Throughput Limitations**
A well-tuned PostgreSQL instance on good (not great) hardware can handle (these assume PostgreSQL's default transaction [[Isolation]] level, [[Read Committed]], where transactions only see data committed before their query began):
- Simple inserts: ~5,000 per second per core
- Updates with index modifications: ~1,000-2,000 per second per core
- Complex transactions (multiple tables/indexes): Hundreds per second
- Bulk operations: Tens of thousands of rows per second

What affects these limits?
- Hardware: Write throughput is often bottlenecked by disk I/O for the WAL
- Indexes: Each additional index reduces write throughput
- Replication: If configured, synchronous replication adds latency as we wait for replicas to confirm
- Transaction Complexity: More tables or indexes touched = slower transactions

**==Write Performance Optimizations==**
- While a single node can handle 5k wps, if we exceed that , what can we do?

1. **Vertical Scaling**: Upgrade our hardware, using faster NVMe disks for better WAL performance, more RAM to increase the buffer cache size, upgrading to CPUs with more cores to handle parallel operations more effectively.
2. **Batch Processing**: The simplest optimization is to batch writes together. Instead of processing each write individually, we collect multiple and execute them in a single transaction (e.g. writing 1000 comments at a time). This means we're buffering writes in our server's memory before committing them to disk (WAL); **==Durability Risk==**: if we crash in the middle of a batch, we lose all the writes in that batch.
3. **Write Offloading**: Some writes don't need to happen synchronously. Analytics data, activity logs, or aggregated metrics can often be processed asynchronously! Instead of writing to Postgres, we can write to:
	- A message queue like [[Kafka]]
	- Have background workers process these queued writes in batches
	- Optionally maintain a separate analytics database.
	- ==This is a good pattern for handling activity logging/analytics events/metrics aggregation/other non-critical updates like "last seen" timestamps==.
4. **Table Partitioning**: For large tables, [[Table Partitioning]] can improve both read and write performance by **splitting data across multiple physical tables**. The most common use case is ==time-based partitioning==.
	- If we had a posts table growing by the millions of rows per month, we might partition that table!
	- Then, different ==database sessions== can write to different partitions simultaneously, increasing concurrency. Also, when data is inserted, index updates only need to happen on th relevant partition, rather than on the entire table. Also, bulk loading operations can be performed partition by partition, making it easier to load large amount of data efficiently.
	- It also just helps with reads! When users view recent posts, PostgreSQL only needs to scan the recent partitions; no need to wade through years of historical data!
		- ==You might even keep recent partitions on fast storage like NVMe drives, while moving other partitions to cheaper storage.==
5. **Sharding**: [[Sharding]] you database is the most common solution in an interview; If a single node isn't enough, Sharding lets you distribute writes across multiple PostgreSQL instances; you'll just want to be clear about what you're sharding on and how you're distributing the data.
	- We might consider ==sharding our posts by user_id so that all the data fora user lives on a single shard==. This is useful to avoid [[Scatter-Gather]] cross-shard queries where we'd have to get data from multiple shards.
	- ==**You typically want to shard on the column that you're querying by the most often!**==
	- **==NOTE:==** Sharding inevitably adds complexity! You'll need to:
		- Handle cross-shard queries
		- Maintain consistent schemas across shards
		- Manage multiple databases
	- Only introduce it when simpler optimizations aren't sufficient!
	- ==Unlike [[DynamoDB]], [[PostgreSQL|Postgres]] doesn't have a built-in sharding solution!==
		- You'll have to implement sharding manually, or you can use managed services like [[Citus]] which handle many of the sharding complexities for you!

```sql
CREATE TABLE posts (
    id SERIAL,
    user_id INT,
    content TEXT,
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create partitions by month
CREATE TABLE posts_2024_01 PARTITION OF posts
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```
Above: Example of **Table Partitioning**


## (3/4) Replication
- Most real-world deployments use [[Replication]] for two key purposes:
	- Scaling reads by distributing queries across the replicas
	- Providing high [[Availability]] in case of node failures
- PostgreSQL supports two types of replication:
	- **==Synchronous==**: Primary waits for acknowledgement from the replicas before confirming the write to the client. ==Stronger consistency but higher latency.==
	- **==Asynchronous==**: Primary confirms the write to the client immediately (after writing on itself) and replicates changes to replicas in the background. ==Potential inconsistency between replicas.==
- **==NOTE:==** Many organizations use a hybrid approach, where a small number of synchronous replicas are used for strong consistency, while maintaining additional asynchronous replicas for read scaling!

**Scaling Reads**
- The most common use for replication is to scale read performance.
- By creating read replicas, you can distribute read queries across multiple DB instances, while still sending all writes to the primary node. This is effective because most applications are read heavy anyways.
	- This multiplies our read throughput by N!
- **==Key Caveat==**: Replication Lag. If a user makes a change and immediately tries to read it back, they might not see their change if they hit a replica that hasn't caught up yet, violating [[Read-your-Writes Consistency]]. (This is only the case for asynchronous replication, I believe; I think synchronous replication should be strongly consistent, which includes the weaker RYW consistency.)


**High Availability**
- By having copies of our data across multiple nodes, we can handle hardware failure without downtime! If our primary node fails, one of the replicas can be promoted to become the new primary.
- This ==failover process== typically involves:
		- Detecting that the primary is down
		- Promoting a replica to primary
		- Updating connection information
		- Repointing applications to the new primary.

==**In your interview, emphasize that replication isn't just about scaling reads, it's also about reliability!**==
- "We'll use replication not just to increase read throughput by distributing reads across the replicas, but also because replication allows our service to stay available even if we lose a database node."

==Most teams use a managed PostgreSQL service like [[AWS RDS]]==, which handle the complexities of failover automatically.

## (4/4) Data Consistency
- If you've chosen to prioritize consistency over availability, then PostgreSQL is a strong choice, built from the ground-up to provide strong consistency guarantees through [[ACID]] Transactions.
- But choosing Postgres isn't enough!
	- ==**WARN:**== A common mistake is to say "WE'll use Postgres because it's ACID compliant", without being able to explain HOW you'll actually USE those ACID properties to solve your consistency requirements.

##### Transactions
- One of the most common points of discussion in interviews
	- ==A [[Transaction]] is a set of operations that are executed together and must either all succeed or fail together.==

Let's consider a simple example where we need to transfer money between two bank accounts:
- Obviously, we want to make sure that if we deduct money from one account, we add it to the other account!

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```
This transaction assures [[Atomicity]]!
- ==Things get more complicated when multiple transactions are happening concurrently==!

##### Transactions and Concurrent Operations











