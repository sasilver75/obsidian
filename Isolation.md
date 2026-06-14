---
aliases:
  - Isolation Level
---
The transaction property that controls how much concurrent [[Transaction]]s can affect eachother before they commit. Defines what a transaction is allowed to see while other transactions are running at the same time.
- Higher isolation makes concurrent transactions behave more like they ran one at a time, preventing  [[Read Phenomenon|Read Phenomena]] like ... but also costs more blocking, validation, retries, or latency.
- Lower isolation gives more concurrency, but allows for more "surprising" interleavings of operations.


The [[ACID]] property that asks:
> "If multiple transactions run at the same time, what unsafe interference are they allowed to have?"


These are the four isolation level from the SQL standard, in descending strictness:
1. [[Serializable Isolation]]: Concurrent transactions must behave as if they executed one at a time, in some serial order.
	- Prevents [[Dirty Read]]
	- Prevents [[Non-Repeatable Read]]
	- Prevents [[Phantom Read]]
2. [[Repeatable Read Isolation]]: Once a transaction reads a row, repeating that same row read must return the same values, but predicate queries may see different committed database states.
	- Prevents [[Dirty Read]]
	- Prevents [[Non-Repeatable Read]]
	- Allows [[Phantom Read]]
3. [[Read Committed Isolation]]: Each read sees only committed data, but different statements within the same transaction may see different committed database states.
	- Prevents [[Dirty Read]]
	- Allows [[Non-Repeatable Read]]
	- Allows [[Phantom Read]]
4. [[Read Uncommitted Isolation]]: A transaction may read data written by other transactions before those transactions commit, so dirty reads, non-repeatable reads, and phantom reads are allowed.
	- Allows [[Dirty Read]]
	- Allows [[Non-Repeatable Read]]
	- Allows [[Phantom Read]]

But other isolation level names are used by real database systems (the standard's four labels aren't expressive enough to describe modern concurrency control precisely):
1. [[Snapshot Isolation]]
2. [[Read Committed Snapshot Isolation]]
3. [[Serializable Snapshot Isolation]]
4. Cursor Stability
5. Read Stability
6. Strict Serializable models, in some distributed databases.

