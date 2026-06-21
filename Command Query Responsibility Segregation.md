---
aliases:
  - CQRS
---
An architectural pattern where the code path and often the data model for ***changing state*** is separated from the code path and data model for ***reading state***.
- A command says: "Please change something"
	- On the command side, the system cares about correctness: validation, permissions, business rules, invariants, transactions.
		- "An order cannot be cancelled after shipment!"
- A query says: "Please tell me something"
	- On the query side, the system cares about convenient retrieval: [[Denormalization|Denormalized]] views, fast lookups, [[Full-Text Search Index|Search Index]]es, reporting tables, [[Cache]]d summaries, and UI-specific shapes.


A more elaborate CQRS system typically might use separate storage:
```
Command side -> canonical write database
              -> emits event / message
              
Query side   -> read model / projection / search index
```
For example, an e-commerce system may write orders into normalized relational tables, but maintain a separate OrderListView table for the customer account page:
```
Order write model:
- orders
- order_items
- payments
- shipments

Order read model:
- customer_order_history_view
  - order_id
  - placed_at
  - status_label
  - total_price
  - thumbnail_url
  - estimated_delivery
```
The read model might even duplicate data intentionally, because its job is not to be the most [[Normalization|Normalized]] representation, its job is to answer the read queries effectively.

==CQRS is most useful when reads and writes have very different needs:==
- A banking ledger write path might need to enforce strict transaction rules, while reporting screens may need precomputed monthly aggregates.

The downside is ==complexity.==
- CQRS introduces more moving parts: separate handlers, duplicated models, synchfronization, stale reads, projection rebuilds, and more tests.


