---
aliases:
  - OCC
---
A way to handle concurrent writes in a database by ==assuming that conflicts/contention are rare,== letting work proceed ==without locking,== and checking at commit/update time whether someone else changed the data first.
- +: Avoid long-held locks, ==works well for read-heavy systems with low contention,== and prevents lost updates.
- -: Under high contention, writers might repeatedly fail and retry; in these cases [[Pessimistic Concurrency Control|Pessimistic Locking]], queues, partitioning, or *single-writer patterns* might be better.

Example: Two users edit the same order.
1. Both read order version 7
2. User A submits an update
3. The database updates the order only if it is still version 7, then increments it to 8.
4. User B submits an update based on their latest read of version 7
5. The database sees the order is now version 8, so ==User B's update *fails with a conflict.==*

Typical shape:
```sql
UPDATE orders
  SET status = 'confirmed',
      version = version + 1
  WHERE id = 'order_123'
    AND version = 7;
```
And then the app checks affected rows:
```
- 1 row updated: success
- 0 rows updated: conflict; reload and retry/merge/reject
```




![[Pasted image 20250518123007.png|600]]