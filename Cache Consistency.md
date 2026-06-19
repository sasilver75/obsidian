
The property that cached copies of data remain synchronized with the authoritative source of truth, according to a system's required correctness guarantee. 
- It doesn't always mean that the cache is instantly up to date, it means that the cache returns data that is fresh enough for the promised model, such as [[Strong Consistency]], [[Read-your-Writes Consistency]], [[Eventual Consistency]], etc. 
- In practice, cache consistency is maintained with techniques like expiration, invalidation, refresh, [[Write-Through Cache|Write-Through]] caching, or version checks.


More plainly: If a value changes in the database, object store, service, or memory location, cache consistency can answer the question:
> When, how, and for whom must the cached copy reflect the change?





