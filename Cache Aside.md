A cache-aside cache is updated after the data is requested. A write-through cache is updated immediately when the primary database is updated. With both approaches, the application is essentially managing what data is being cached and for how long.

A cache-aside cache is the most common caching strategy available. The fundamental data retrieval logic can be summarized as follows:
1. When your application needs to read data from the database, it checks the cache first to determine whether the data is available.
2. If the data is available (_a cache hit_), the cached data is returned, and the response is issued to the caller.
3. If the data isn’t available (_a cache miss_), the database is queried for the data (by the application). The cache is then populated with the data that is retrieved from the database, and the data is returned to the caller.