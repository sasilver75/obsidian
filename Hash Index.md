![[Pasted image 20250520180212.png]]

Hash Indexes are just a hashmap that maps indexed values to row locations.
- The database maintains an array of buckets, where each bucket can store multiple key-location pairs.
- When indexing a value, the database hashes it to determine which bucketshould store the pointer to the row data.
- Hash collisions are handled through **linear probing**, meaning worst-case lookups can degrade for O(N), but for a good **hash function** and **load factor**, we'll typically see O(1).

This makes Hash Indexes ==incredibly fast for **exact-match queries**==.
- Compute the hash, go to the bucket, follow the pointer

This same structure ==makes them useless for **range queries or sorting**==, since similar values are deliberately scattered across different buckets.

Real-World Usage:
- Despite their speed for exact matches, ==hash indexes are relatively rare in practice, with people instead choosing to use [[B-Tree]]s==.
- PostgreSQL supports them but doesn't use them by default, since B-Trees perform nearly as well for exact-matches while supporting range queries and sorting.
- Still, ==Hash indices do shine in specific scenarios, particularly for in-memory databases!== [[Redis]] uses hash tables as its primary data structure for KV lookups because all data lives in memory.
	- When using ==disk-based storage==, [[B-Tree]]s are usually the better choice due to their efficient handling of disk I/O patterns.

  
**In your system design interviews, you might consider hash indexes when:**
- You need the ==absolute fastest possible exact-match lookups==
- You'll ==never need range queries or sorting==    
- You have ==plenty of memory== (hash indexes tend to be larger than B-trees)


Downside: Your keyset generally needs to fit in memory, and there's poor support for range queries.