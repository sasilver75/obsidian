---
aliases:
  - Streaming Replication
---
[[Physical Replication]] ships the [[Write-Ahead Log|WAL]] byte-for-byte, whereas [[Logical Replication]] decodes the WAL into row-level changes.

Physical Replication: Typically for same (e.g.) [[PostgreSQL|Postgres]] version, whole cluster, read-only replicas.
Logical Replication: For cross-version, selective tables, writable subscribers, usable for [[Change Data Capture|CDC]]/Realtime.





