---
aliases:
  - Crash Recovery
  - Point-in-Time Recovery
---

I'm using Database Recovery to refer to the set of mechanisms that a database management system sues to return the database to a correct and up-to-date state after a failure.

I'm specifically referring to:
1. ==Crash Recovery==: When a database process or server loses power or crashes, and is recovered with write-ahead logs, redo, undo.
2. ==Point-in-time Recovery==: When someone accidentally deletes rows. Restore backup, replay logs to a chosen time.

Database often want to preserve two transaction guarantees:
- Atomicity: A transaction's changes are all applied, or non are applied.
- Durability: Once a transaction commits, the transaction's effects survive crashes.

The hard part is that databases do not write every data change to disk immediately; they often cache data pages in memory for performance. 

Think of the database as having two durable things:
1. The data files: These contain the actual tables, indexes, pages, rows, and metadata. The data files may lag behind, or even contain partial changes depending on when the crash happened.
2. The transaction log: This records what changes transactions made, in an ordered queue. These are the database's memory of what really happened.

After a crash, the DB asks:
- What transactions *definitely* committed?
- What transactions were active, but did not commit?
- Which committed changes might be missing from the data files?
- Which uncommitted changes might have reached the data files?

Then, recovery does two broad actions:
- Redo: Reapply committed changes that might be missing from data files.
- Undo: Remove uncommitted changes that may have reached data files.

> Redo winners, undo losers. Committed transactions should be reapplied, while uncommitted, interrupted transactions are rolled back.


# Why does the [[Write-Ahead Log]] (WAL) exist?
- Without a log, the database would have two bad choices:
	1. Write every changed page to disk, before commit. 
		- This makes commits slow, because a single transaction might touch many random disk pages.
		- Instead, databases often use a [[Buffer Cache]]/BufferPool, an in-memory cache of disk pages. Transactions modify pages in memory (as well as a quick-to-update WAL), and *later,* the database flushes dirty [[Page]]s back to disk.
	2. Allow commits *before* all data pages are written, but have no durable record of those changes.
		- This risk the loss of data during a crash.

The transaction log solves this, because write logs are usually sequential and much cheaper than random data-page writes. Instead of saying *"Before commit, write every changed table and index to disk,"* the database says *"Before commit, write enough log records to disk so that the committed changes can be reconstructed later."*

There are two key rules to WALing
1. Before a dirty data page is written to disk, the log entry record describing changes on that page must already be durable. This ensures that if the data page reaches disk with uncommitted changes, recovery knows how to undo those changes.
2. Before the database reports a transaction as committed, the transaction's commit log record must be durable. This ensures that if the database tells the client "commit succeeded," recovery can redo the transaction if changed data pages were not written before the crash.

These rules let the DB decouple transaction commit from writing all modified data pages.




# ARIES-Style Recovery Flow
- Many database systems use ideas from [[Algorithms for Recovrety and Isolation Exploiting Semantics|ARIES]], a classic WAL recovery algorithm. Specifics might differ from system to system, but the high-level shape is common:

1. ==Analysis==: Figure out what was happening at the time of the crash.
	- The database starts from the most recent checkpoint and scans forward through the log, reconstructing which transactions were active, which pages may have unwritten/dirty changes, and the commit status of transactions. 
	- At the end of this phase, the database *knows* which committed transactions might need redo, which uncommitted transactions might need undo, and approximately where redo must begin.
2. ==Redo==: Reapply changes that might not have reached the data files.
	- This means that the database reapplies logged changes, *including changes from transactions that later need to be undone*.
	- Each page can store the Log Sequence Number (LSN) of the latest change applied to that page. During redo, if the page already has that change, the database skips it. This makes redo [[Idempotency|Idempotent]]: applying redo twice has the same effect as applying it once.
3. ==Undo==: Roll back transactions that did not commit.
	- The database rolls back transactions that were active but not yet committed when the crash happened.
	- These undo operations themselves must be logged, otherwise a crash recovery could leave the database in another ambiguous state: Many systems write compensation log records. This way, if the database crashes again during recovery, the *next* recovery attempts doesn't incorrectly undo the same operation *twice.*


Q: Why is it the case that we have to also be able to undo? Why are uncommitted changes every written to data pages on disk?
A: So the database usually modifies data pages in memory first; these modified in-memory page are called dirty pages. Eventually, dirty pages have to be written back to the data files on disk. So why/when does a database write a dirty page before the transaction that dirtied the page commits? There are some major policy choices that a database must make, along two axes:
- Force: At commit, force all changed data pages to disk.
- No-Force: At commit, do not require all changed data pages to be written to disk.
- Steal: Allow pages with uncommitted changes to be written to disk.
- No-Steal: Never write pages with uncommitted changes to disk.
The common high-performance design in databases is
```
steal + no-force
```
This means:
1. Uncommitted changes may reach disk (`steal` policy)
	- This happens when the database flushes a dirty page to disk before the transaction that modified that page commits. The database might need buffer-pool space for other queries, or maybe a background writer/checkpoint wants to flush dirty pages.
2. Committed changes may not yet be in data files, at commit time (`no-force` policy)
	- This happens when the database commits by forcing the transaction *log* to disk, but does not immediately force every changed data *page* to disk.
This design gives good performance, but it requires *both* undo and redo operations in crash recovery.


# [[Checkpoint]]s
- Without checkpoints, crash recovery might need to scan an enormous log.
- A checkpoint records enough information to reduce recovery time. After a crash, the database can usually start recovery near the last checkpoint, and proceed the recovery process from there.
	- Note that in many systems, checkpoints are [[Checkpoint|Fuzzy Checkpoint]]s, meaning normal transaction processing continues while the checkpoint is taken, meaning all data might not be safely written.

