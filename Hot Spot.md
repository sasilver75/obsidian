---
aliases:
  - Hot Key
  - Hot Partition
---
A hot key in a cache is a cache key that receives a disproportionately large share of traffic. For example, if a Redis cluster stores keys like:
```
user:123
user:456
post:999
homepage:global_feed
```
If `homepage:global_feed` is requested 100k times per second, that's a hot key!
- If you have a distributed cache, a given key is usually mapped to one cache shard.
- So even if the cache cluster has 50 nodes, all the requests for that one key may hammer the same node.
- The result can be high latency, CPU saturation, network saturation, cache timeouts, or even a [[Cache Stampede]] when that hot cache key expires.


Common Fixes:
- Replicate the hot key across multiple cache nodes.
- Add local [[In-Process Cache]]s to the requesting services.
- Regarding expiration of hot keys:
	- Use [[Request Coalescing]].
	- Use [[Refresh-Ahead]].

1. Detect hot keys using per-key access metrics
2. Protect the origin/database with [[Request Coalescing]] or [[Stale-While-Revalidate]] or [[Refresh-Ahead]] behavior.
3. Reduce pressure on the cache shard with [[In-Process Cache]]s or replicated hot-key storage.
4. Redesign the key/data shape, if the hot key is really a symptom of one global object being too central.


# How does the "Replicate the hot key across multiple cache nodes" work?

So we want to take a situation like this:
```
one logical key -> one physical key -> one cache node
```
And turn it into this:
```
one logical key -> many physical keys -> many cache nodes
```
It's a read-scaling technique.
The cost is more complicated writes, invalidation, freshness guarantees, and expiry behavior.
For highly read-heavy data, it's worth it!

How do we do it?
If `product:123` receives 80,000 reads per second, some node is absorbing almost all of that traffic.
We need to create multiple physical keys:
```
product:123:replica:0 -> hashes to node 2
product:123:replica:1 -> hashes to node 8
product:123:replica:2 -> hashes to node 13
product:123:replica:3 -> hashes to node 21
product:123:replica:4 -> hashes to node 34
```
Now, the 80,000 rps are spread across five cache nodes, so each sees 16,000 reads per second.

Writes now become more expensive because the system has to update or invalidate every replica.
If `product:123` changes price, the application *may* (depending on caching pattern, e.g. [[Write-Through Cache|Write-Through]]) need to do this:
```
SET product:123:replica:0 new_value
SET product:123:replica:1 new_value
SET product:123:replica:2 new_value
SET product:123:replica:3 new_value
SET product:123:replica:4 new_value
```

This replication improves scalability, but *can*  introduce a consistency problem, where different replicas may temporarily hold different values:
```
product:123:replica:0 = price $10
product:123:replica:1 = price $10
product:123:replica:2 = price $12
product:123:replica:3 = price $12
```
This might be find for a trending article count, a public feed, or a recommendation list, but might be unacceptable for account balance, authorization state, inventory reservation, or payment status.

Q: How do we actually select a replica, as a client?
1. Client-side replica selection, where the application knows there are 10 replicas and chooses one before calling Redis or Memcached:
```python
replica_index = math.random(0,9)
cache_key = f"homepage_feed:replica:{replica}"
```

2. You could also do it with *request-stable selection*, where instead of choosing randomly every time, the application chooses the replica based on something stable, like user ID:
```
replica_index = hash(user_id) % replica_count
```
This spreads users across replicas, but one user tends to read the same replica repeatedly. This can improve locality and reduce confusing cases where one user sees value A, then value B, then value A again.

3. A third variant is local in-process replication, sometimes called a near cache. Each application server keeps its own short-lived copy of the hot value in memory, which reduces load on the distributed cache itself.
```
request -> application-local memory cache -> Redis/Memcached -> database
```
This is especially useful for very hot, small, mostly read-only values such as feature flags, public configuration, exchange rates, or homepage modules.


failure Modes
- The biggest failure mode is a ==replica stampede==, where all replicas expire at the same time, so many requests may miss at once and recompute the same value.
- This is why hot-key replication is usually paired with:
	- [[Time to Live|TTL]] jitter
	- [[Request Coalescing]]
	- Soft TTLs
	- [[Stale-While-Revalidate]]
	- Background refresh (e.g. [[Refresh-Ahead]])

For example, instead of giving all replicas a 60-second TTL, give each replica a slightly different TTL:
```
replica 0: 55 seconds
replica 1: 63 seconds
replica 2: 71 seconds
replica 3: 58 seconds
replica 4: 66 seconds
```
This avoids a synchronized expiration, causing a replica stampede (a form of a [[Cache Stampede]])





