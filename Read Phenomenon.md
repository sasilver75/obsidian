---
aliases:
  - Read Phenomena
  - Serialization Anomaly
  - Serialization Anomalies
  - Concurrency Anomaly
  - Concurrency Anomalies
---
An [[Isolation]]-related [[Read Phenomenon|Read Phenomena]]

Common Read Phenomena
- [[Dirty Write]]: A transaction overwrites data written by another uncommitted transaction; very dangerous, and prevented by most databases even at their weaker isolation levels.
- [[Dirty Read]]: A transaction reads another transaction's uncommitted data. If the writer rolls back, the reader saw that that never officially existed, which can subsequently effect the data it writes.
- [[Non-Repeatable Read]] (Fuzzy Read): A transaction reads the same row twice, and get different values because another transaction committed an update between reads.
- [[Read Skew]] (Inconsistent Read): A transaction reads multiple related rows, but they come from different logical points in time.
	- e.g. Reading account A before a transfer and account B after a transfer, making the total money look wrong.
- [[Phantom Read]]:  A transaction reruns a predicate query and gets a different set of rows.
	- `SELECT count(*) FROM bookings WHERE room_id = 5` returns `0`, then another transaction inserts a booking, then the same query returns `1`.
	- c.f. Non-Repeatable Read
		- In NRR: The same row changes between reads.
		- In PR: The same predicate/query result set changes between reads. Conceptually, it's like a NRR over a set of rows, rather than one previously-read row. Locking existing rows isn't enough, you need predicate/range locks, serializable validation, exclusion constraints, etc.
	- c.f. Read Skew
		- In RS: About reading an inconsistent snapshot across related data.
		- In PR: About rerunning the same predicate and getting a different result set because matching rows were inserted/updated/deleted.

These two are ==NOT== really read phenomena, but are broader isolation-related anomalies that I'll include here:
- [[Lost Update]]: Two transactions read the same value, compute a new value, and one overwrites the other.
	- e.g. Two writers both read `counter=10` and both write `counter=11`; one increment 
- [[Write Skew]]: Two transactions read overlapping data, each makes a valid local update, but together they violate an invariant.
	- e.g. Two doctors both go off call after seeing that the other is still on-call.


Could be grouped as:
- Dirty/intermediate visibility anomalies: [[Dirty Write]], [[Dirty Read]]
- Changing-read anomalies: [[Non-Repeatable Read]], [[Read Skew]], [[Phantom Read]]
- Write conflict anomalies: [[Lost Update]], [[Write Skew]]


