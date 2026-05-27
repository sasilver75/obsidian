---
aliases:
  - Pessimistic Locking
  - PCC
---
Method for managing simultaneous access to shared data that assumes that conflicts are likely, so it locks data before allowing a transaction to read or modify it, preventing other users or processes from changing the same record while one transaction is working on it.

Example:
1. Transaction A opens an order record for update.
2. The database locks that record.
3. Transaction B tries to update the same order.
4. Transaction B must wait, fail, or time out until Transaction A commits or rolls back.


+: Prevents lost updates, and good when conflicts are common.
-: Can reduce throughput in scenarios with less contention, can lead to [[Deadlock]]s, requires careful transaction design.



cf. with [[Optimistic Concurrency Control]] (OCC)