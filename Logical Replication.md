[[Logical Replication]] decodes the WAL into row-level changes, whereas [[Physical Replication]] ships the [[Write-Ahead Log|WAL]] byte-for-byte.
- Physical Replication: Typically for same (e.g.) [[PostgreSQL|Postgres]] version, whole cluster, read-only replicas.
	- Use physical replication when you want a full copy of the primary for high availability, disaster recovery, read scaling, and straightforward failover.  
- Logical Replication: For cross-version, selective tables, writable subscribers, usable for [[Change Data Capture|CDC]]/Realtime.
	- Use logical replication when you want to replicate selected data, migrate across versions, feed change data capture pipelines, move data into other systems, customize the target schema/indexes/partitions, or combine multiple sources into one target.

Logical replication copies *the meaning of the data changes,* such as "insert this row," or "update this column," so that ==another database can reproduce selected data changes without being storage identical.==

e.g.
```
In table accounts, update row with primary key id = 42:
set balance = 900
```
In PostgreSQL, logical replication is usually built on logical decoding, where Postgres reads the write-ahead log and decodes physical WAL records into logical row-change records. Those events are then sent from a publisher to a subscriber.

For a SQL statement like:
```sql
UPDATE users
SET email = 'new@example.com'
WHERE id = 123;
```
In Logical replication, the subscriber would receive a logical change resembling:
```
table: users
operation: UPDATE
key: id = 123
new values: email = 'new@example.com'
```

Logical replication is perhaps better than physical replication when the target database doesn't need to be a full physical copy.
- Copying only *some* tables
- Feeding an analytics database
- Implementing [[Change Data Capture]]
- Consolidating data from multiple databases
- Replication into a differently-indexed or partitioned target.

The downside is that logical replication usually requires more explicit management. Schema changes, permissions, sequences, replication identity, code handling, and initial synchronization matter more.
- In Postgres, built-in logical operations mainly replicate [[Data Manipulation Language|DML]] changes: `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`. It does not generally replicate arbitrary schema changes like `CREATE TABLE`, `ALTER TABLE`, or `CREATE INDEX` in the same automatic way physical replication does. The subscriber usually needs compatible schema already present.

Misconception: “Logical replication is always better because it is more flexible.”
- Logical replication is more flexible, but physical replication is usually simpler and stronger for exact standby replicas and failover.

