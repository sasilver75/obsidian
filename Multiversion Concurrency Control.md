---
aliases:
  - MVCC
---
A database technique where updates create new versions of rows, rather than immediately overwriting the old version. This is a way to get higher concurrency while lettering readers and writers avoid blocking each other.

>"Keeping multiple versions of data so that each transaction can read a consistent version without blocking writers."

> "The database keeps multiple committed/uncommitted versions of rows, and each transaction reads the version visible to its snapshot."

Mainly about improving concurrency:
- Usually avoids "Reader blocks writer"
- Usually avoids "Writer blocks reader"
- Avoids [[Dirty Read]]s, because readers ignore any uncommitted versions, as well as versions later than their snapshot.
- Consistent statement/transaction view is achieved using snapshots.

Without MVCC, a writer might do:
```
row=100
writer locks row
writer changes row to 150
reader must wait, because the row is being changed
```
With MVCC:
```
old committed version: row=100
writer creates new uncommitted version: row=150
reader can still read old version: row=100

When the writer commits, future reads can see 150, depending on their transaction snapshot.
```

# How it Works
- Transactions read from a ==snapshot==
- A snapshot answers: Which transaction had committed when this read/transaction began?
- When scanning rows, the DB chooses the row version visible to that snapshot.

```
Initial:
balance=100

Transaction A (T1) starts updating:
A creates a new version: balance=150
A has not committed yet

Transaction B (T2) reads:
B's snapshot does not include T1
B sees old version: balance=100

A commits

Transaction C (T3) starts:
C's snapshot includes T1
C sees new version: balance=150

~~~~~ Account Row Versions ~~~~~~
v1: balance=100, created_by=T0, deleted/replaced_by=T1
v2: balance=150, created_by=T1
```
Eventually, when no active transactions can still see `v1`, the databases can clean it up.

### Snapshots
- A snapshot can be scoped differently, depending on [[Isolation]] level:
	- [[Read Committed Isolation]]: Each statement gets a fresh snapshot
	- [[Repeatable Read Isolation]]: Whole transaction uses one snapshot
	- [[Serializable Isolation]]: Database also prevents outcomes that could not happen in a serial order.


# MVCC Tradeoffs
MVCC gives better concurrency, but has both benefits and costs!
- Benefits 
	- Readers usually don't block writers
	- Writers usually don't block readers
	- Reads get consistent snapshots
	- Avoids [[Dirty Read]]s
- Costs
	- Old row versions must be stored
	- Cleanup/[[Vaccuum]]/garbage collection is needed.
	- Long-running transactions can keep old versions alive.
	- More complex internals and visibility rules.




_____________

MVCC is a versioning-based concurrency control mechanism

While the names might make you think that it's similar in some sense to [[Optimistic Concurrency Control]] (OCC) or [[Pessimistic Concurrency Control]] (PCC), MVCC is actually somewhat on a different axis.

[[Optimistic Concurrency Control|OCC]] and [[Pessimistic Concurrency Control|PCC]] relate to Conflict Strategy: in PCC, we block before conflict, while in OCC, we allow work, but validate later.

In contrast, MVCC is a Data Visibility/Storage Strategy. It defines how version s are stored and read. MVCC can be combined with either style.