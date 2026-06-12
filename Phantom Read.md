An [[Isolation]]-related [[Read Phenomenon|Read Phenomena]]

When a transaction returns the same predicate query and sees a different set of rows because another committed transaction inserted/deleted/updated rows that now no longer match the query condition.
- Mechanically, this is because the database does not fully protect the gap, range, or predicate that defines which rows could appear. Locking the 10 existing pending orders doesn't prevent another transaction from inserting an 11th pending order, unless the database also locks the relevant range or uses a serializable mechanism.

Example:
```sql
-- Transaction A
BEGIN;
SELECT COUNT(*) FROM orders WHERE status = 'pending';
-- returns 10

-- Transaction B
BEGIN;
INSERT INTO orders (status) VALUES ('pending');
COMMIT;

-- Transaction A, same transaction
SELECT COUNT(*) FROM orders WHERE status = 'pending';
-- returns 11
```
- The extra row seen in the second Transaction A query is called a **phantom*** because it wasn't visible in the first query, but appears when the Transaction repeats the same logical query.
- A Phantom read can also happen from an `UPDATE` or `DELETE`


Common Prevention Mechanisms:
- [[Serializable Isolation]], which makes the concurrent transactions behave as if they ran one at a time.
- [[Predicate Lock]]s, which protect the logical condition being queried.
- [[Key-Range Lock]]s, which protect index ranges where matching rows could appear.
- [[Multiversion Concurrency Control|MVCC]] Snapshots, which make a repeated read in the same transaction see the same committed snapshot, even though this doesn't always prevent every serializability anomaly.


# Comparison with other [[Read Phenomenon|Read Phenomena]]:
- [[Dirty Read]]: You read uncommitted data.
	- Transaction A reads a row that Transaction B hasn't committed yet.
- [[Non-Repeatable Read]]: The same existing row changes.
	- Transaction A reads account balance `$100`, later reads same row as `$80`
- [[Phantom Read]]: The matching row set changes.
	- Transaction A counts 10 pending orders, later counts 11.

