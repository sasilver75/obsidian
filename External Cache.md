
A cache stored outside the application process, usually in a separate service or infrastructure layer.

It differs from a a local in-memory cache ([[In-Process Cache]] ) because multiple app instances can share it, and cached data can survive individual app restarts. The tradeoff is that each cache access usually has network latency and operational complexity.

Examples:
- [[Redis]]
- [[Memcached]]
- [[Content Delivery Network|CDN]]


c.f. [[Client-Side Cache]], [[In-Process Cache]]