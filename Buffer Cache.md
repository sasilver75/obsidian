---
aliases:
  - Buffer Pool
---
Memory owned and managed by a database engine, which caches database [[Page]]s, fixed-size chunks of table and index files, such as 8KiB [[PostgreSQL|Postgres]] pages or 16 KiB [[MySQL]] InnoDB pages.

The database buffer pool knows database-specific facts:
- Which database page is cached
- Whether the page is dirty
- Which transactions are using or pinning the page
- Which log sequence number the page reflects
- Whether the page can be evicted
- Whether the page must wait for [[Write-Ahead Log|WAL]] records before being flushed.

It's a performance ache, but it's also part of the storage engine's correctness machinery, because it interacts with WAL logging, [[Checkpoint]], dirty-page flushing, and [[Database Recovery|Crash Recovery]].