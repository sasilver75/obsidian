Efficient storage and retrieval for time series data.
Mostly large numbers of inserts from different sources that are stored in the order in which they're written.

Optimizations:
- Column-oriented storage (and compression)
- Smart caching (of a source + time range combination at a time)

