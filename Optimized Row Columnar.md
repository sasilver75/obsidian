---
aliases:
  - ORC
---
A ==columnar== binary serialization format.
- Designed for the [[Hadoop]] ecosystem ([[Apache Hive]], [[Apache Spark|Spark]])
- Excellent compression and predicate pushdown (can skip entire row groups if a filter doesn't match)
- ==Good for analytical, read-heavy workloads==.

