---
aliases:
  - Outbox
  - Outbox Pattern
---

A pattern that ensures reliable, at-least-once message and event delivery in distributed systems to resolve the [[Dual Write Problem]] (where (e.g.) a system updates a local database but fails to publish a message to a broker) by guaranteeing that both actions succeed or fail together [[Atomicity|Atomically]].
- You could use a [[Two-Phase Commit]] (2PC), but it's possible that the (e.g.) database or message broker might not support 2PC.

Problem: How to atomically update the database *and* (e.g.) send messages to a message broker?

Solution: ==The service that sends the message to *store the broker-bound message in the database as part of the transaction that's already updating business entities!* A separate process then *sends* the messages to the message broker.==
- This implies [[Eventual Consistency]]
![[Pasted image 20260604101725.png]]
Above:
- See that the Order service developers *want* to atomically update both the database and a message queue.
- Instead of writing directly to both in some way, the Order Service bundles both of the writes into a single transaction on the database, writing to the `ORDERS` table as well as an `OUTBOX` table.
- Later, some other process ("Message Relay") asynchronously processes the Outbox table records and inserts records into the queue.


# Comparison with [[Change Data Capture]] (CDC)
- CDC typically trails the Database's [[Write-Ahead Log]] (WAL)/binlog and filters for inserts to (e.g.) the `ORDERS` table, and then updates the search index. The CDC connector keeps a long-lived connection to this log stream and reads new committed records as they appear, so it's typically pretty low latency (though you can also set your polling short for an Outbox+Message Relay version)
- Comparison
	- Transactional Outbox means the app explicitly writes an event as part of the same DB transaction as the business change. Important property is semantic intent: "OrderPlaced" or "InvoicePaid" exists as a durable event, because the app chose to create it.
	- In CDC, an external process watches the DB commit log and reacts to row changes. The important property is observation: "this row in `orders` changed was noticed after commit."
	- ==So the outbox is better when downstream systems need stable domain events. CDC is better when downstream systems just need a projection of database state, as is the case for search indexes, analytics, caches, or replicas.==