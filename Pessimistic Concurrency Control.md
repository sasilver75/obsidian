---
aliases:
  - Pessimistic Locking
  - PCC
---
Method for managing simultaneous access to shared data that assumes that conflicts are likely, so it [[Lock]]s data before allowing a transaction to read or modify it, preventing other users or processes from changing the same record while one transaction is working on it.
- +: Useful when conflicts are common/contention is high, and when retrying work would be expensive. 
- -: Reduced concurrency, waiting/blocking, deadlocks if locks are taken in an inconsistent order.
	- Reduces concurrency because it makes other work wait. The cost of a lock has two parts: bookkeeping (small), and lock holding time. While one transaction holds the lock, conflicting transactions cannot proceed.

Example:
1. Transaction A opens an order record for update.
2. The database locks that record.
3. Transaction B tries to update the same order.
4. Transaction B must wait, fail, or time out until Transaction A commits or rolls back.


+: Prevents lost updates, and good when conflicts are common.
-: Can reduce throughput in scenarios with less contention, can lead to [[Deadlock]]s, requires careful transaction design.



cf. with [[Optimistic Concurrency Control]] (OCC)
- OCC: No lock first; fail if the version changed from what you expected.
- PCC: Lock first, make others wait.


Imagine a profile editing system where users mostly edit different rows, and conflicts are rare:
With OCC
```
With OCC:
User A reads profile 1 version 3
User B reads profile 2 version 9
User C reads profile 3 version 5
```
All submit updates independently, and most succeed. No one waits for locks on read, the DB only checks versions at update time.

OCC helps when the work before a commit is relatively long, but conflicts are uncommon.
Example:
- Read object
- Call pricing service
- Compute eligibility
- Prepare update
- Commit if version unchanged

Under OCC, many requests can do that work in parallel, while under PCC, if they lock first, conflicting requests wait during the external call/computation.

Bu if many requests fight over the same row, OCC can be worse:
```
100 workers all update the same inventory counter
99 fail version check
retry
fail again
retry...
```
In this case, PCC, a queue, atomic decrement, or a single-writer approach many outperform OCC.


> "PCC reduces concurrency because it blocks contenders early and for the duration of the transaction. OCC improves throughput when conflicts are rare because it lets the work proceed in parallel, and only rejects the few stale writes at the end."