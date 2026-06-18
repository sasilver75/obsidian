


# Comparison with [[Checkpoint]]
- Checkpoint main purpose is to make [[Database Recovery|Crash Recovery]] faster, capturing recovery metadata, and often flushed dirty pages.
	- "At this log position, enough database state is known or flushed that recovery does not need to start from the beginning of the log."
- Snapshot main purpose is to preserve or expose a point-in-time view; a consistent view/copy of data at some moment. Used for reads, backups, cloning, rollback, and [[Database Recovery|Point-in-Time Recovery]].
	- "A view of something as it existed at a particular time," but the exact meaning depends on context.
	- Common uses:
		- Transaction snapshot: A consistent read used by concurrency control, e.g. [[Snapshot Isolation]], used for isolation. Under [[Multiversion Concurrency Control|MVCC]], a transaction can read the database as it looked when the transaction began.
		- Storage snapshot: A [[Copy-on-Write]] or block-level point-in-time image of files/volume, used for backup/cloning/rollback. e.g. the files as of `02:00`.
		- Backup snapshot: A consistent backup image of the database at a time, which can serve as the starting point for point-in-time recovery.

