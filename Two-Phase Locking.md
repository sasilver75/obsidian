---
aliases:
  - 2PL
---
A database concurrency control protocol where each transaction has two lock phases:
- A **growing phase**: Where the transaction may acquire locks but may not release any.
- A **shrinking phase**: Where the transaction may release locks, but may not acquire any new ones.

The strict guarantee:
> If every [[Transaction]] follows Two-Phase Locking, the resulting scheduled is conflict-serializable. This means that the interleaved execution behaves like the transactions ran one-at-a-time in some serial order.

> If every transaction follows Two-Phase Locking over every relevant data item, predicate, or key range, then 2PL guarantees conflict-serializable execution, which is sufficient for serializable isolation.
- But there are common caveats:
	- Basic 2PL isn't what databases usually mean by practical serializable isolation, because basic 2PL may release write locks before commit, which can allow [[Dirty Read]] and cascading aborts. Real systems using locking for serializable isolation usually use strict Two-Phase Locking, where exclusive locks are held until commit or abort (plus [[Predicate Lock]]s as necessary).
	- The database must lock the right things. Row-level 2PL alone may not prevent [[Phantom Read]]s.

A transaction is allowed to do this:
```
Acquire, acquire, acquire, acquire...
Release, release, release...
```
But not this:
```
Acquire, acquire, release, acquire
```
Once a transaction releases its first lock, the transaction has crossed from the growing phase into the shrinking phase. After that point, the transaction may only release locks.


# Example
Suppose two transactions touch objects `A` and `B`:
```
T1:
  read A
  write B

T2:
  read B
  write A
```
A valid 2PL execution might be:
```
T1 acquires S(A)
T1 reads A
T1 acquires X(B)
T1 writes B
T1 releases S(A)
T1 releases X(B)

T2 acquires S(B)
T2 reads B
T2 acquires X(A)
T2 writes A
T2 releases S(B)
T2 releases X(A)
```
In this scenario, T1 acquired all needed locks before releasing any lock. T2 did the same. The result is equivalent to the serial ordering: `T1, then T2`.

