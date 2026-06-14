---
aliases:
  - Fuzzy Read
---
An [[Isolation]]-related [[Read Phenomenon|Read Phenomena]]

A database concurrency anomaly where one transaction reads the same row twice and sees different committed values, because another transaction modified or delete that row between the two reads.

> "I read this row earlier in my transaction, but when I read it again, it had changed!"

Typically, the flow resulting in a non-repeatable read looks like:
1. A transaction reads a row.
2. Another transaction updates or deletes that same row.
3. The other transaction *commits*.
4. The first transaction reads the same row again, seeing a different committed result.

A Non-Repeatable Read *matters* when application code assumes that values remain stable during a transaction.

This anomaly is allowed under the following isolation levels:
- [[Read Uncommitted Isolation]]
- [[Read Committed Isolation]]
This anomaly is *prevented* under the following isolation levels:
- [[Repeatable Read Isolation]]
- [[Serializable Isolation]]

# Comparison with [[Dirty Read]]
The important detail is that the second value is *committed.*
- If instead the first transaction were reading *uncommitted* data from the second transaction, that would be a [[Dirty Read]], not a Non-Repeatable Read.

# Comparison with [[Phantom Read]]
- A non-repeatable read is about an existing *row* changing.
- A phantom read is about a query returning *a different set of rows*.


# How do databases prevent it?
1. [[Lock]]s: DB prevents other transactions from modifying rows that the current transaction has read until the current transaction finishes.
2. [[Multiversion Concurrency Control]]: Database keeps multiple committed versions of rows. A transaction reads from a stable snapshot instead of always reading the last committed version.
3. [[Serializable Isolation|Serializable]] execution: Database ensures that the final result is equivalent to some one-at-a-time ordering.


# Example
```
Starting State: 
user1: Balance 100
```
Transaction A runs:
```sql
BEGIN;

SELECT balance
FROM accounts
WHERE id = 1;
-- returns 100
```
Meanwhile, Transaction B runs:
```sql
BEGIN;

UPDATE accounts
SET balance = 50
WHERE id = 1;

COMMIT;
```
Back in Transaction A:
```sql
SELECT balance
FROM accounts
WHERE id = 1;
-- returns 50
```

So... Transaction A saw `100` on its first read, but `50` on its second read. 
This is a non-repeatable read. It's "non-repeatable" because repeating the same read inside the same transaction did not return the same result.



