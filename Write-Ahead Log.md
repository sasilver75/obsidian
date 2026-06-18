---
aliases:
  - WAL
---
An append-only recovery log that records a database change *before the database is allowed to rely on the corresponding data-file change being durable.*

The problem it solves is ==crash safety==. Updating database pages in place is risky: The process, machine, disk or OS can fail halfway through a write. A WAL gives the database a durable, ordered record of what was supposed to happen, so that after a crash the database can reconstruct a consistent state.
- Before moving money between accounts, the database writes the intended change into a durable notebook. If the database crashes halfway through rearranging its actual storage pages, recovery reads the notebook and either finishes the committed work or rolls back incomplete work.


Mechanically:
1. A transaction modifies data.
2. The database creates one or more WAL records describing the change.
3. Those WAL records are appended sequentially to the WAL.
4. Before the transaction is acked to the client as committed, the relevant WAL records, including the commit record, must be flushed to durable storage.
5. The actual table/index pages may be written later!
6. If the database crashes, recovery starts from a recent checkpoint/snapshot, and replays WAL records that must be redone, and removes effects from transactions that did not commit.

The "Write-Ahead" part:
> The log record describing a data-page change must reach durable storage before the changed data page itself is written durably. The database writes the log first, then writes the data.


The exact types of records that are in a AWAL are database-specific, but conceptually WAL records usually fall into these groups:
- Data modification records: Row/page changes from INSERT, UPDATE, DELETE
- Index modification records: Changes to indexes, such as [[B-Tree]] page inserts, deletes, or splits
- Transaction control records: COMMIT, ABORT/rollback, prepared transaction records
- Checkpoint records: A recovery starting marker
- Storage/catalog records: Create/drop/truncate/extend table or index files
- Full-page images: A whole copy of a database page, often after the first change following a checkpoint
- Compensation/undo-related records: Records saying “this earlier change was undone”
- Replication/logical records: Logical messages or changes decoded for replicas/subscribers

A simplified example:
```sql
BEGIN;

UPDATE accounts
SET balance = balance - 100
WHERE id = 'A';

UPDATE accounts
SET balance = balance + 100
WHERE id = 'B';

COMMIT;
```
A simplified WAL might look like this:
```
LSN 100: BEGIN transaction 42

LSN 110: UPDATE transaction 42
         table = accounts
         row = A
         before balance = 500
         after balance = 400

LSN 120: UPDATE transaction 42
         table = accounts
         row = B
         before balance = 200
         after balance = 300

LSN 130: COMMIT transaction 42
```
- LSN means "==Log Sequence Number==," the position of a record in the WAL.
- Now imagine a CRASH
	- If the database crashes after LSN120 but before LSN130, recovery sees two updates but does not see a commit record. 
		- The database treats transaction 42 as incomplete. Depending on the database architecture, recovery either undoes those effects, ignores those row versions, or marks the transaction aborted.
	- If the database were to instead crash after LSN130, recovery sees the commit record. The database must preserve the transfer. If the table pages on disk don't yet show the final balances, recovery replays the WAL records and restores the committed sate.



# Physical, Logical, and Physiological logging
There are several styles of logging:
- Physical logging: Log describes byte-level or page-level changes ("On page 12, change bytes 80-90")
	- Can be forward to redo/undo in [[Database Recovery|Crash Recovery]], but can be verbose.
- Logical logging: High-level operation ("Insert row with id 5 into table users")
	- Can be compact and flexible, but redo must be careful because the logical operation may not be repeatable in exactly the same environment.
- Physiological logging: Logical operation within a physical page ("On page 12, insert this record")
	- A common compromise between physical/logical.
