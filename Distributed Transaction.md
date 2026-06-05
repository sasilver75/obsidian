[[Transaction]]s that ==span more than one independent system,== database, service, or resource manager, while trying to preserve transaction guarantees like [[Atomicity]].
- Distributed transactions are expensive and failure-prone, so many architectures prefer weaker consistency patterns unless strict atomicity is required.

Example: Transferring money between two banks:
```
1. Debit $100 from ACcount A in Bank DB 1
2. Credit $100 to Account B in Bank DB 2
3. The system must avoid ending up with only the debit or only the credit.
```
Because multiple systems are involved, coordination is required. 


A common classic approach is the [[Two-Phase Commit]] (2PC):
1. Prepare Phase: Coordinator asks each participant: "Can you commit?" Participants start local transactions but don't yet commit.
2. Commit Phase: If everyone says yes, coordinator tells all to commit, otherwise tells them to roll back. After a participant says "Yes," it *cannot abort*. If the coordinator crashes, the participant is stuck in doubt.
The hardest part is failure handling (networks failing, coordinator or participants crashing, locks being held too long). Coordinator failure or network partition turn into blocked resources across multiple systems, and not all data systems support the 2PC protocol.


In modern distributed systems, ==teams often avoid *classic* distributed transactions when possible==, using alternatives like:
- [[Saga]]s: Break work into steps with [[Compensating Action]]s. Still provides eventual atomicity, but breaks work into multiple committed local transactions, then uses compensating actions if a later step fails.
- [[Transactional Outbox Pattern|Outbox Pattern]]: Commit local DB changes and secondary intent (e.g. message publishing intent) together. 
- [[Idempotency|Idempotency Key]]s: Making retries safe
- [[Eventual Consistency]]: Accepting temporary inconsistencies while systems converge.





