
A database [[Lock]] on a logical condition., such as `WHERE status='pending'`, rather than on specific rows.
It prevents other transactions from inserting, updating, or deleting rows in a way that would change the set of rows matching that condition.

It is a means of solving the [[Phantom Read]] problem.
```sql
-- Transaction A
BEGIN;
SELECT COUNT(*) FROM orders WHERE status = 'pending';
-- returns 10

-- Transaction B
BEGIN;
INSERT INTO orders (status) VALUES ('pending');
COMMIT;

-- Transaction A
SELECT COUNT(*) FROM orders WHERE status = 'pending';
-- without protection, this could now return 11! That would be a phantom read.
```

Mental model:
> A predicate Lock locks the *question,* not just the current answers. If the question is "Which orders are pending?", the lock protects the set of possible rows that could affect the answer.

Mechanically, when Transaction A runs:
```sql
BEGIN;
SELECT * FROM orders where status = 'pending';
...
```
The database records that Transaction A depends on the predicate `status = 'pending'`.
If Transaction B tries to insert or update a row so that the row satisfies `stauts= 'pending'`, that write conflicts with Transaction A's Predicate Lock.

Q: This seems like it would really really limit concurrency, if a read is blocking writes!
A: Yes, that's right. Pessimistic Predicate Locking can heavily limit concurrency, especially if the predicate is broad, unindexed, or long-lived. This is why *==real databases typically AVOID pure arbitrary predicate locks!==* Instead, they use narrower or more optimistic techniques:
- [[Key-Range Lock]]s: Lock a range of ordered index key values, not an arbitrary logical condition.
- [[Multiversion Concurrency Control|MVCC]] Snapshots: Let readers and writers proceed without blocking eachother, but provide weaker guarantees unless combined with serializable checks.
- Serializable conflict detection: Allow concurrency, then abort a transaction if the combined read/write pattern would not be serializable.
- Coarser fallback locks: If the DB can't represent a predicate precisely, it may lock a page, relation, or table, which is safer but worse for concurrency.






