---
aliases:
  - Streaming Replication
---
[[Physical Replication]] ships the [[Write-Ahead Log|WAL]] byte-for-byte, whereas [[Logical Replication]] decodes the WAL into row-level changes.
- Physical Replication: Typically for same (e.g.) [[PostgreSQL|Postgres]] version, whole cluster, read-only replicas.
	- Use physical replication when you want a full copy of the primary for high availability, disaster recovery, read scaling, and straightforward failover.  
- Logical Replication: For cross-version, selective tables, writable subscribers, usable for [[Change Data Capture|CDC]]/Realtime.
	- Use logical replication when you want to replicate selected data, migrate across versions, feed change data capture pipelines, move data into other systems, customize the target schema/indexes/partitions, or combine multiple sources into one target.

Physical replication copies the database's low-level storage changes so that another server becomes an almost byte-for-byte copy of the primary. ==Typically requires that both share the same storage system.==

Usually includes everything in the replicated database cluster:
- Table data
- Indexes
- System catalogs
- Transactions tate
- Visibility information
- Internal storage layout
- Most database-level state needed for crash recovery and consistency

Is especially good for [[Failover]]/ high availability scenarios because the standby is a complete copy of the primary. If the primary fails, the standby can be promoted.
- The tradeoff is inflexibility; the standby is usually not an independently-shaped databases, it's a replica of the primary (this is usually fine).


```
PRIMARY DATABASE
----------------

Application runs:

  UPDATE accounts
  SET balance = balance - 100
  WHERE id = 42;

PostgreSQL writes physical WAL records:

  WAL record 1: modify heap/table page 1842
  WAL record 2: modify index page 391
  WAL record 3: mark transaction 781233 as committed

        stream WAL records
              |
              v

STANDBY DATABASE
----------------

Standby receives WAL records:

  replay WAL record 1: apply table-page change
  replay WAL record 2: apply index-page change
  replay WAL record 3: mark transaction committed

Result:

  standby database files now match the primary's database files
```
- The key idea: physical replication does **not** send “update account 42” as the main abstraction. It sends lower-level write-ahead log records that let the standby reproduce the primary’s storage changes.

