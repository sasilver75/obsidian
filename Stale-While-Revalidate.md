A cache serving policy where an ==expired cached value may still be returned immediately for a limited stale window, while a background refresh fetches and stores a fresh value for later requests==.
- Avoid if you can't tolerate "stale" data, since SWR can serve stale data from the cache.

Verbal usage:
- "Cached for 15 seconds, stale for 5 minutes while revalidating" means:
	- For 15 seconds, the cached response is fresh
	- After 15 seconds, it is technically safe
	- For the *next* 300 seconds, the cache may still serve the stale response immediately, while kicking off an asynchronous job in the background to repopulate the value for the cache key.
	- In a CDN, this might be: `Cache-Control: public, max-age=0, s-maxage=15, stale-while-revalidate=300`


Stale While Revalidate:
```
Cache hit and still fresh:
-> Return cached value

Cache hit and stale-but-servable:
-> Cached value is stale/expired, but still within an allowed stale window
-> Trigger background refresh
-> Return cached value
-> Next request gets refreshed value

Cache hit and stale-and-unservable:
-> Cached value is stale/expired, and is outside the allowed staleness window
-> Block, and fetch/recompute a fresh value
-> Store that value in the cache
-> Return cached value

Cache miss:
-> Block, and fetch/recompute a fresh value
-> Store that value in the cache
-> Return cached value
```

So you might have two time windows:
- Fresh for 60s
- Stale-but-Servable for another 300s

So if data is 30s old, return it normally
If data is 90s old, return it immediately but refresh in background
If data is 500s old, it is too stale; block and fetch refresh data.

==Useful for when fast responses matter more than *perfect* freshness (pages, feeds, product listings, dashboards, CDN caches), but risky for balances, permissions, inventory checkout, or anything where stale data causes correctness problems.==

In HTTP/[[Content Delivery Network|CDN]] caching, this can be expressed with headers like:
- `Cache-Control: max-age=60, stale-wrhile-revalidate=300`
In an app cache, you implement the same logic in code.


Similar in a sense to [[Refresh-Ahead]], in that both use background refresh to avoid making reads wait, the difference is when the refresh is triggered:
- In ==Refresh-Ahead==, it's more proactive; the cache decides to refresh before expiry, often based on TTL, access frequency, or scheduled refresh.
- In ==Stale-while Revalidate== is more reactive: a read request finds expired-but-servable data, returns it, and triggers a refresh.