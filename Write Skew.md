An [[Isolation]]-related anomaly.

A concurrency anomaly where two transactions each read some shared state, make a decision that seems valid, and then write to different records, but the combined result violates an invariant.

> When concurrent transactions read overlapping data but write non-overlapping data, so that ordinary write-write conflict detection doesn't catch the conflict.

Classic doctor example, with the invariant: "At least one doctor must be on call"
```sql
-- Transaction 1
BEGIN;

SELECT COUNT(*)
FROM doctors
WHERE shift_id = 10
  AND on_call = true;

-- Result: 2
-- Transaction 1 concludes: "It is safe for Alice to go off call,
-- because Bob is still on call."

UPDATE doctors
SET on_call = false
WHERE shift_id = 10
  AND doctor = 'Alice';

COMMIT;
```
Meanwhile:
```sql
-- Transaction 2
BEGIN;

SELECT COUNT(*)
FROM doctors
WHERE shift_id = 10
  AND on_call = true;

-- Result: 2
-- Transaction 2 concludes: "It is safe for Bob to go off call,
-- because Alice is still on call."

UPDATE doctors
SET on_call = false
WHERE shift_id = 10
  AND doctor = 'Bob';

COMMIT;
```
- Each transaction was locally reasonable; each saw another doctor still on call. But together, the transactions broke the invariant.  
	- The key subtlety is that the transaction did not update the same row; T1 wrote to Alice's row, wihle T2 wrote to Bob's row... so a database isolation level that only detects direct write-write conflicts might allow both transactions to commit.

Write skew is especially associated with [[Snapshot Isolation]], when each transaction reads from a consistent snapshot of the database as of the transaction start time.
- While Snapshot Isolation usually prevents [[Dirty Read]]s, [[Non-Repeatable Read]]s, and many [[Lost Update]]s, it can still allow write skew because both transactions can read from the same old snapshot, then write different rows.

The deeper issue is that the conflict isn't located in one row, the conflict lives in an invariant, such as:
- "At least one doctor must be on call."
- "A meeting room must not have overlapping reservations."
- "A user may have at most one active primary email address."
- "Total allocated budget must not exceed the department budget."

> Write skew is what happens when each transaction asks, “Is my change safe?” against an old snapshot, but no transaction asks, “Are all concurrent changes safe together?”

To prevent write skew, you usually need one of these:
1. [[Serializable Isolation]]: The database ensures that the result is equivalent to some one-at-a-time ordering of transactions.
2. Explicit Locking: In this case, lock all rows involved in the invariant before deciding.
3. Database Constraints: When the invariant can be expressed as a unique constraint, exclusion constraint, foreign key, check constraints, or similar mechanism.
4. Materialize the invariant into a single conflict pair: For example, store a counter or guard row that every transaction must update, causing a write-write conflict.

