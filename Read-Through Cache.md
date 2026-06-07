---
aliases:
  - Read-Through
---
A Cache pattern where the application reads from the cache, and the cache layer itself handles cache misses by loading data from the backing store.
- [[Redis]]/[[Memcached]] themselves do not usually know how to fetch this data from your database; Read-through requires a cache library/framework/custom cache service with configured loader.

Flow:
```
application asks cache for key X
	-> if cache hit, return value
	-> if cache miss
		cache fetches value from database/origin
		cache stores value
		cache returns value
```
Interestingly, ==the application does not manually query the database on a cache miss== (as in a [[Cache-Aside]] pattern); instead, that logic is hidden behind the cache abstraction.



![[Pasted image 20260607103404.png]]







