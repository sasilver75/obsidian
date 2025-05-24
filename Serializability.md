


It appears to the writer as if all transactions ran on a single thread.


Ways to achieve:
- Actually running in a single thread, like VoltDB
- [[Pessimistic Concurrency Control]]: Hold locks on all read-from and written-to DB rows to prevent concurrent updates. Better when we have a lot of contention in our transactions, so that we're blocking rather than looping <run whole transaction> and <roll back> like in OCC.
- [[Optimistic Concurrency Control]]: Allow transactions to proceed with no locks and abort any transactions that see the data that they've read or written to has been modified. Nice when there isn't much contention.