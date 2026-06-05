---
aliases:
  - Isolation Level
---
The transaction property that controls how much concurrent [[Transaction]]s can affect eachother before they commit. Defines what a transaction is allowed to see while other transactions are running at the same time.
- Higher isolation makes concurrent transactions behave more like they ran one at a time, preventing  [[Read Phenomenon|Read Phenomena]] like ... but also costs more blocking, validation, retries, or latency.
- Lower isolation gives more concurrency, but allows for more "surprising" interleavings of operations.


The [[ACID]] property that asks:
> "If multiple transactions run at the same time, what unsafe interference are they allowed to have?"








Ordering maybe not right, not complete
[[Serializable Isolation]]
[[Read Committed Isolation]]
[[Read Uncommitted Isolation]]
[[Repeatable Read Isolation]]
