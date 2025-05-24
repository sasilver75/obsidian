
Reading from disk is slow. We can introduce in-memory caches, which whold a subset of our data, to speed up data access.

Like databases, caches can and often should use [[Replication]] and [[Sharding]]


We need to consider:
- Read Policy (What do we do when the Cache misses? Does the Cache hit the backing datastore )
- Write Policy (What do we do when we want to write data?)
	- [[Write-Through Cache]]:  BOTH the cache and the underlying datastore** **simultaneously**! Ensures that your cache is **consistent** with your backing datastore, but can be **slower** for write operations since you have to do something like [[Two-Phase Commit|2PC]]
	- [[Write-Around Cache]]: Data only written to DB; Data is pulled into the cache on a miss. Faster write and DB is source of truth, but Cache misses can be expensive.
	- [[Write-Back Cache]]: An application writes directly to the Cache, and at some point the Cache asynchronously writes to Database. Super fast, and if you aren't reading from the Cache, you might not see the latest writes, plus list of data loss.
- [[Cache Eviction Strategy]]
- [[Cache Invalidation Strategy]]


Geographically distributed caches for static content are called [[Content Delivery Network]].