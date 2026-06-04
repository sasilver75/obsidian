A cache refresh strategy where ==frequently-used cache entires are proactively refreshed before they expire, usually in the background==, so later reads can keep hitting fresh cached data without waiting for a reload.

It trades extra background work for fewer cache misses and lower read latency.
- ==Use when a small set of known hot keys are read often, are expensive to recompute, and when you want to avoid users ever hitting the cache miss and experiencing the latency of a refresh.==

This is a policy that has to be implemented by application code, a cache library, a worker, etc. To refresh a cache entry, *something* must know both the cache key and the loader function for rebuilding it. Redis itself may know that `product:123` expires soon, but it doesn't know how to fetch product 123 from your DB and rebuild the cached JSON.

A common implementation is read-triggered:
- read `product:123`
- cache hit
- entry is still fresh, but expires in 5 seconds
- return cached value immediately
- enqueue background refresh for product:123. which loads fresh data from the DB/API and writes a new cache value with a new TTL.
(Pretty similar to [[Stale-While-Revalidate]], mechanically)

Another implementation is worker-driven:
- A job periodically refreshes known hot keys:
	- Every minute:
		- Get top dashboard/product/feed keys
		- Refresh keys expiring soon
This seems more like what you'd expect, but of course this one doesn't seem to have a knowledge of when keys are going to expire.

_________

Similar in a sense to [[Stale-While-Revalidate]], in that both use background refresh to avoid making reads wait, the difference is when the refresh is triggered:
- In ==Refresh-Ahead==, it's more proactive; the cache decides to refresh before expiry, often based on TTL, access frequency, or scheduled refresh.
- In ==Stale-while Revalidate== is more reactive: a read request finds expired-but-servable data, returns it, and triggers a refresh.

Comparison to [[Read-Through Cache]]:
- Refresh-Ahead proactively populates the cache with data from the backing store *before* it is explicitly required.
- Read-Through Caches fetch data from the backing store only when it is explicitly requested by the application.

