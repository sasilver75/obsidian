---
aliases:
  - Cache Write Strategies
---

- [[Write-Through Cache|Write-Through]]: Write to Redis and Postgres before acknowledging success. The cache is updated synchronously with the durable store, so that reads after the write usually see fresh cached data. It's nice to get fresh reads, but writes are slower because the backing database write is on the critical path.
```
Client sends update
  -> application validates request
  -> application writes new value to Postgres
  -> application commits Postgres transaction
  -> application writes corresponding value to Redis
  -> application returns success
```

- [[Write-Around Cache|Write-Around]]: Write only to Postgres; do not update Redis. Redis is populated later, typically on cache miss, by whatever strategy is being used. Writes go *around* the cache; Redis is updated later only if the future read needs the value. This means avoids filling Redis with data that might never be read. This might imply that following reads don't see the new data, if your application doesn't also invalidate the old redis key. 
	- Q: If you *do* choose to take the time to delete/invalidate the old Redis key, why not also write the new value? You're already "there" at Redis?
		- A: The reason is to avoid filling the cache with values that were written but may never be read. A value being updated does not prove that the value is worth caching.
```
Client sends update
  -> application validates request
  -> application writes new value to Postgres
  -> application commits Postgres transaction
  -> application does not write the new value to Redis
  -> application may delete/invalidate the old Redis key
  -> application returns success
```

- [[Write-Back Cache|Write-Back]] (Write-Behind): Write to Redis, with Postgres later being updated asynchronously with flushes from the cache. The cache temporarily becomes the write buffer; gives fast writes, but introduces durability and consistency risks. Writes complete at the cache first, and the backing store is updated later. Until the background write succeeds, Redis contains newer data than Postgres.
```
Client sends update
  -> application validates request
  -> application writes new value to Redis
  -> application marks the value as dirty or enqueues a database-write job
  -> application returns success
  -> background worker later writes new value to Postgres
  -> background worker commits Postgres transaction
  -> background worker marks the Redis value/job as flushed
```
Later, during the flush:
```
Background worker finds pending write
  -> background worker reads pending value from Redis or a queue
  -> background worker writes value to Postgres
  -> background worker commits Postgres transaction
  -> background worker clears dirty marker / acknowledges queue message
```
