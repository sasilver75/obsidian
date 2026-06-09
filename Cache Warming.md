
The practice of preloading a [[Cache]] with data before users or systems request it, so the first real requests are faster.

Instead of waiting for a [[Cache Miss]] and then fetching or computing the data on demand, the system proactively fills the cache ahead of time.

Examples:
- After deploying a website, we can pre-render popular pages into a [[Content Delivery Network|CDN]] cache
- Before a  sales dashboard opens on Monday, we preload common reports into a Cache.
- When an app starts, load frequently used database records an [[In-Process Cache]]

The goal is to reduce cold-start latency, avoid spikes from many simultaneous cache misses, and make performance more predictable.

Tradeoff: Cache warming uses extra compute, memory, and sometimes stale-data risk if the warmed data changes before it's read.




