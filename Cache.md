---
aliases:
  - Caching
---
Reading from disk is slow. We can introduce in-memory caches, which whold a subset of our data, to speed up data access.

Like databases, caches can and often should use [[Replication]] and [[Sharding]]


We need to consider:
- [[Cache Read Strategy]] (What do we do when the Cache misses? Does the Cache hit the backing datastore )
	- [[Read-Through Cache|Read-Through]]
	- [[Cache-Aside]]
- [[Cache Write Strategy]] (What do we do when we want to write data?)
	- [[Write-Through Cache]]:  BOTH the cache and the underlying datastore** **simultaneously**! Ensures that your cache is **consistent** with your backing datastore, but can be **slower** for write operations since you have to do something like [[Two-Phase Commit|2PC]]
	- [[Write-Around Cache]]: Data only written to DB; Data is pulled into the cache on a miss. Faster write and DB is source of truth, but Cache misses can be expensive.
	- [[Write-Back Cache]]: An application writes directly to the Cache, and at some point the Cache asynchronously writes to Database. Super fast, and if you aren't reading from the Cache, you might not see the latest writes, plus list of data loss.
- [[Cache Eviction Strategy]]
- [[Cache Invalidation Strategy]]
- [[Cache Warming]]
- [[Hot Spot|Hot Key]]s
- [[Cache Stampede]]s
- [[Content Delivery Network]]
- [[Cache Hit]]
- [[Cache Miss]]

_____________________
SDIAH

A cache is just a temporary storage that keeps recently  used data handy so that you can fetch it quickly next time.

![[Pasted image 20260605194514.png]]

So where should you cache your data?


One option is an [[External Cache]], which is where the cache (e.g. [[Redis]], [[Memcached]]) runs on its own server, totally separate from your application.
![[Pasted image 20260605194617.png]]
- [[Cache-Aside]]: When your application needs data, it first checks the Cache. If it's not there, it's a [[Cache Miss]], so it fetches the data, stores it in the cache, and then returns it to the user.


> SAM: CONTINUE HERE 


