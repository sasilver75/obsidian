A distributed cache is a cache spread across multiple servers, so many application instances can share fast access to the same cached data.

Examples include [[Redis]] Cluster, [[Memcached]], Hazelcast, or managed cloud caches.

Relative to a normal (implied single-instance) [[External Cache]], Distribution allows us to increase all of capacity, write-throughput, and read-throughput, depending on the configuration... but comes with the operational complexity and things like [[Failover]], [[Eventual Consistency]], possible data loss, etc.




