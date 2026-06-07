---
aliases:
  - Locking
  - Mutex
  - Semaphore
  - Shared Lock
  - Exclusive Lock
  - Read Lock
  - Write Lock
---
A database mechanism that temporarily restricts what other [[Transaction]]s can do to the same resource, to prevent two transactions from making incompatible changes at the same time. 
- The main mechanism for[[Pessimistic Concurrency Control]], where conflicts are prevented by making transactions acquire permission before accessing shared data.

> "Before doing an operation on some resource (row, table, index, ...), a transaction must first acquire permission."

#### Lock Types:
- ==Shared Lock==/Read Lock: "I'm reading this. Other reads may also read, but writers must wait."
- ==Exclusive Lock==/Write Lock: "I'm changing this. Nobody else may read or write in a conflicting way until I'm done."

#### Lock Granularities (non-exhaustive, contextual):
- Row Lock: Lock one record, e.g. when updating one account.
- Pack Lock: Locks a storage [[Page]] containing many rows, some databases do this internally.
- Table Lock: Lock the whole table, e.g. when bulk loading, doing schema changes, full-table updates.
- Database Lock: Locks a large database-level object, e.g. an administrative operation.

Tradeoff: Smaller locks allow for more productive concurrency, but cost more book-keeping, while larger locks are simpler, but block more work.

#### Common Operations:
- Read data: May acquire shared/read locks, depending on DB/[[Isolation]] level.
- Update/delete row: Acquires exclusive/write lock on that row.
- Insert row:
- Change table schema:
- Enforce uniqueness:


Generic Locking Example:
```
account,balance = 100

Transaction A wants to subtract 30
Transaction B wants to subtract 50

If both read 100 and both write back independently:
A writes 70
B writes 50

One update will be lost!

...

With locking, though:

A locks account row
A reads 100
A write 70
A commits
A releases lock

B locks account row
B reads 70
B writes 20
B commits
B releases lock

```
The lock forces the critical section to happen one transaction at a time.




# What happens when a query is trying to read data that has an exclusive lock on it?
- It depends on the database and the kind of read!
- For a normal read:
	- May wait, or may read an older committed version
- For a locking read/write:
	- Usually waits for the exclusive lock to release

In an [[Multiversion Concurrency Control|MVCC]] database (e.g. [[PostgreSQL|Postgres]]), normal reads don't usually have to wait on a row-level write lock.

But otherwise, most databases have a lock manager, tracking:
- resource: accounts row id=1
- current lock holder: transaction A
- lock mode: exclusive/write
- wait queue: transaction B, transaction C, ...

If another operation requests an incompatible lock:
1. DB checks lock compatibility
2. Sees exclusive lock already held
3. Puts requesting transaction into a wait queue
4. Blocks/suspends that operation
5. When transaction A commits or rolls back, A releases lock
6. DB wakes one or more waiters
7. Waiting transaction retries/acquires lock

If the wait is too long, you may get:
- lock timeout
- statement timeout
- deadlock detected




