---
aliases:
  - Block Range Index
---
A [[PostgreSQL|Postgres]] Block Range Index is a specialized, compact index designed for massive tables with ==naturally ordered data==, such as timestamps or sequential IDs.
- Ideal for large, append-only datasets like logs, time-series, or sensor data.

By storing only minimum and maximum values for ranges of physical pages, BRIN indexes are often ==thousands of times smaller than [[B-Tree]]s== and significantly speed up range queries on data that correlates with physical disk storage.

A BRIN is a "lossy" index; it narrows down the search to specific block ranges, after which the database rechecks the actual rows.


```sql
-- Create a BRIN index
CREATE INDEX idx_logs_time ON logs USING BRIN (log_time);

-- Create a BRIN index with custom range size
CREATE INDEX idx_logs_time_small ON logs USING BRIN (log_time) WITH (pages_per_range = 32);
```

