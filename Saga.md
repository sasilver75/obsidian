---
aliases:
  - Sagas
---


A way to handle [[Distributed Transaction]]s that are an alternative to tradition ACID distributed transactions like [[Two-Phase Commit]]. 

A saga breaks a [[Distributed Transaction]] into a sequence of smaller, independent local [[Transaction]]s.  
- If a step in the middle of a saga fails, the saga executes a series of [[Compensating Action]]s (undo operations) to revert changes made by preceding steps. 
- Because the intermediate steps are committed immediately, the system can temporarily be in a state of [[Logical Consistency|Logical Inconsistency]]. The Saga's goal is  to ensure that all changes resolve into a consistent state *eventually.*
- Typically "favored" in [[Microservice]] architectures because they improve availability by avoiding the long-lived locks required by (e.g.) [[Two-Phase Commit|2PC]].


## Classic Example: Checkout
Typical order of operations:
1. Create order
2. Reserve inventory
3. Authorize payment
4. Arrange shipment
5. Mark order confirmed

Each step is its own transaction in its own service/database. There is no global lock and no global rollback. If a payment authorization fails after inventory was reserved, the saga runs a compensation like `"release inventory" and "cancel order"`.

## Types of Saga Coordination:
- ==Orchestration==: A central coordinator tells each participant what local transaction to execute, and triggers compensations if something fails.
	- Easier to reason about, debug, and monitor.
- ==Choreography==: Services communicate by publishing and listening to domain events without a central coordinator.
	- Events:
		- OrderCreated -> Inventory reserves stock
		- InventoryReserved -> Payment charges card
		- PaymentCharged -> Shipping schedules shipment
	- There is no central controller, so it becomes hard to trace because control flow is spread across event handlers.


==IMPORTANT:== Sagas ==DO NOT PROVIDE ACID ISOLATION.== Other parts of the system may observe intermediate states:
- Order exists but is not paid
- Inventory is reserved, but payment later fails and that inventory later gets un-reserved.
- Payment succeeds, but shipment scheduling is impossible, so it later gets refunded.

Saga-based systems usually model these states explicitly: `pending`, `reserved`, `confirmed`, `caceling`, `canceled`, `failed`, `needs_manula_review`, etc.

Good Saga design depends heavily on:
- [[Idempotency]]: Every step and compensation must be safe to retry.
- Durable state: The Saga's process must be stored so that it can resume after it crashes.
- Timeouts: If a service doesn't respond, the saga needs a policy.
- Comprensations
- Observability: You need to see stuck, retrying, failed, and compensated sagas
- Manual repair paths: Some failures cannot be perfectly automated

A useful mental model is that a saga is state machine for a business process. The orchestrator moves the process from state to state until it reaches a terminal state like `completed`, `cancelled`, or `failed`.


## Tradeoff
- [[Two-Phase Commit|2PC]] gives stronger atomicity but can block and hold locks across systems.
- [[Saga]]s avoid global locks and tolerate distributed failure better, but expose intermediate states and require careful compensation logic.

 
 Use sagas when the workflow is business-level, multi-service, and long-running.
- ==Avoid them== when you truly need strict atomicity and isolation; in that case, ==keeping the data in one transaction boundary is usually cleaner==.


