---
aliases:
  - LSM Tree
---
References:
- Blog: [How Cassandra Stores Data: An Exploration of Log Structured Merge Trees](https://hackernoon.com/how-cassandra-stores-data-an-exploration-of-log-structured-merge-trees)


A storage engine/data structure ==optimized for high write throughput==.
- Used in systems like [[RocksDB]], [[LevelDB]], [[Apache Cassandra|Cassandra]], [[ScyllaDB]], [[HBase]], [[CockroachDB]], and parts of many modern storage systems.
- The main downsides are:
	- Read amplification: A read might need to consult multiple SSTable files
	- Write amplification: Compact rewrites data in the background
	- Space amplification: Old versions of the data are shadowed and coexist until compaction
	- Compaction cost: Background merging can consume CPU, I/O, and cause latency spikes.

The core idea:
> Instead of updating data in-place on disk, we write new data sequentially, and then merge sorted files in the background.

This makes writes fast because disks, SSDs, and filesystems are generally much better at *Sequential Writes* than *Random Writes*!

1. Writes go to a [[Log]]. When you write `user:123 = Alice`, the database first appends it to a write-ahead log so it can recover after a crash.
2. Writes go into memory. The value is inserted into an in-memory sorted structure, often called a ==Memtable==.
3. Memtable flushes writes to disk. Once the memtable gets large enough, it is written to disk as an immutable sorted file, often called a [[Sorted String Table]] (SSTable).
4. Reads check multiple places. To read a key, the database may check: the current memtable, then the recently-flushed SSTables, then older SSTables. [[Bloom Filter]]s and indexes can help avoid scanning unnecessary files.
5. Compaction merges files. Over time, many SSTables accumulate. Background compaction merges them, discards overwritten values, removes deleted records, and keeps read performance manageable.

Why "Log Structured"?
- Because updates are treated as append-only log entries:
```
put user:123 Alice
put user:456 Bob
put user:123 Alicia
delete user:456
```
- The database doesn't immediately overwrite `user:123`, it just appends the new value.  Later, the [[Compaction]] process decides that `Alice` is obsolete, and keeps only `Alciia`.

Why "Merge Tree?"
- "Merge": Because sorted runs of data ([[Sorted String Table|SSTable]]s) are repeatedly merged together
```
Memory:
A C F

Disk file 1:
B D G

Disk file 2:
A E H

After compaction:
A B C D E F G H
```
- "Tree" refers to a hierarchy of levels, where newer/smaller files are merged into older/larger files.


# Comparison with [[B-Tree]]s
- Both are ways to maintain an ordered key-value index on disk, though in practice people often say [[B-Tree]] to mean an index structure, and [[Log-Structured Merge Tree|LSM Tree]] to mean a whole storage-engine strategy.
	- B-Trees can be either a [[Secondary Index]] or a [[Primary Index]]
	- LSM Trees: Primary storage or a [[Secondary Index]]
- B-Tree mutates sorted pages in-place (dealing with the consequences, in terms of the self-balancing tree)
- LSM-trees append writes to a log and maintain an in-memory tree, and later flush the tree to sorted files, merging with compaction later.

Both can answer:
```
Get key=X
Scan keys from A to Z
Find next key after X
```
But they optimize different costs
