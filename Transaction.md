See also: [[Distributed Transaction]]


A group of operations that's treated as a single [[Atomicity|Atomic]] unit of work, where the whole thing succeeds, or none takes effect.

Transactions are typically described with [[ACID]] properties:
- [[Atomicity]]: All operations happen, or none do.
- [[Logical Consistency]]: The database moves from one valid state to another, according to rules that the database knows about (foreign keys, uniqueness constraints, check constraints, non-null constraints, etc.)
- [[Isolation]]: Concurrent transactions should not interfere in unsafe ways.
- [[Durability]]: Once committed, the result should survive crashes (i.e. is on disk)


Below: a transaction transferring money between two accounts in a database
```sql
  BEGIN;

  UPDATE accounts
  SET balance = balance - 100
  WHERE id = 'A';

  UPDATE accounts
  SET balance = balance + 100
  WHERE id = 'B';

  COMMIT;
```
Above:
- If something fails halfway through, the transaction rolls back, so that Account A is not debited without Account B being credited.



# What determines what locks need to be taken, given a transaction?
- In Postgres, locks are mostly taken statement-by-statement inside the transaction.
- `BEGIN` mainly starts a transaction context, then each SQL statement determines locks based on what it does.

```sql
  BEGIN;
  SELECT * FROM accounts WHERE id = 1;
  COMMIT;
```
Above: In Postgres, this normal select uses [[Multiversion Concurrency Control|MVCC]], taking a lightweight table-level `ACCESS SHARE` lock so that the table is not dropped or radically changed while being read, but it doesn't lock the table it reads.

```sql
  BEGIN;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  COMMIT;
```
Above: An `UPDATE` takes :
- A table-level `ROW EXCLUSIVE` lock on `accounts`
- A row-level lock on the specific row(s) it updates
These row-level locks are what blocks another transaction from updating/deleting the same rows at the same time.


Locks are determined by:
- The statement type (`SELECT`, `INSERT`, `UPDATE`, etc.)
- The explicit locking clauses (`FOR UPDATE`, `FOR NO KEY UPDATE`, `FOR SHARE`, `FOR KEY SHARE`, etc.)
- [[Isolation]] level: Especially `SERIALIZABLE`, which adds predicate/SIREAD-style tracking to detect dangerous concurrent patterns.
- Constraints and indexes (unique checks, foreign keys, exclusion constraints, etc. may take additional locks)
- [[Data Definition Language|DDL]] vs [[Data Manipulation Language|DML]]