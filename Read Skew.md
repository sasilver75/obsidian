---
aliases:
  - Inconsistent Read
---
An [[Isolation]]-related [[Read Phenomenon|Read Phenomena]] 

A transaction reads multiple related rows, but they come from different logical points in time, so the transaction observes a combination of values that never existed together at any committed database state.
- e.g. Reading account A before a transfer and account B after a transfer, making the total money look wrong.

The mental model:
> Under some isolation levels, a transaction doesn't necessarily read from one stable snapshot. Each statement may see "the latest committed data" at the time that the individual statement runs. This means that a multi-query transaction can accidentally stitch together facts from different database snapshots/point in time.

Initially:
```
checking_balance + savings_balance = 1000
checking_balance = 500
savings_balance = 500
```
Transaction R is reading both balances to compute the total
Transaction W transfers 100 from checking to savings:
```
checking_balance = 400
savings_balance = 600
```
A read skew can happen like this:
```
R reads checking_balance = 500

W updates checking_balance to 400
W updates savings_balance to 600
W commits

R reads savings_balance = 600
R computes total = 1100
```
The value `1100` is impossible.


Read Skew is especially associated with (happens under) [[Read Committed Isolation]], which prevents [[Dirty Read]]s (so that that Transactions won't see uncommitted data), but still may see different committed versions of different rows across statements.

It's usually prevented by stronger isolation levels, like: 
- [[Repeatable Read Isolation]]: Re-reads of the same row are stable, and many implementations use a transaction snapshot.
- [[Snapshot Isolation]]: The whole transaction reads from one consistent snapshots.
- [[Serializable Isolation]]: The result must be equivalent to some serial execution.

# Comparison with [[Non-Repeatable Read]]
Read Skew is closely related to a [[Non-Repeatable Read]], but the emphasis is different:
- A non-repeatable read means a transaction reads the same row twice and gets different values.
- Read skew means a transaction reads multiple related values and gets an inconsistent mixture of old and new values.

# Comparison with [[Write Skew]]
- Write skew is a *write anomaly:* two transactions each make a decision from a valid snapshot, then write different rows in a way that jointly violates an invariant.
- Read skew is a read anomaly: A transaction observes an inconsistent state.

