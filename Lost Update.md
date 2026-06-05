An [[Isolation]]-related anomaly.

A problem that happens when two actors read the same old value, both compute or choose a new value, adn one overwrites the other without noticing.

Example:
```
display_name = "Alice"
version = 1
```
Two clients send writes to the leader:
```
A: set display_name = "Alice Smith"
B: "Alice Johnson"
```
The leader receives A first, then B
If there is no concurrency check, both succeed in order:
```
version 2: display_name = "Alice Smith"
version 3: display_name = "Alice Johnson"
```
This is a [[Lost Update]] problem.

==A's update is "lost" because B overwrote it without ever seeing or intentionally replacing it.==


Another class example is a counter:
```
Initial_balance=100
```
Two readers read 100, then:
```
Operation A: add 10 -> write 110
Operation B: add 20 -> write 120
```
Final value becomes `120`, even though the correct value should have bee `130`


# How do we solve it?
- [[Optimistic Concurrency Control]]: Update only if the version you read is still current. Write is rejected if the version the writer read is no longer current.
- [[Pessimistic Concurrency Control]] (Locking): Actors acquire locks. Only one actor can edit/update at a time.
- Atomic Operations: use `increment by 10` instead of `read, compute, write`


