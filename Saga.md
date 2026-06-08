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
- ==Avoid them== when you truly need strict atomicity and isolation; in that case, ==keeping the data in one transactional boundary is usually cleaner== (meaning keep it a [[Monolith]] if you can!)


________________
# IMPORTANT CAVEATS ABOUT ISOLATION AND SAGA DESIGN

Q: What if we have a situation where we have steps A, B, and C that need to be done in a Saga, and we have two active transactions, with the following scenario where both Trx1 and Trx2 involve `user-123` in some way:
```
Trx 1: Does A: Increments user's balance from $40 to $60
Trx 1: Does B: ...
Trx 2: Does A: Decrements user's balance $50, from $60 to $10
Trx 1: Tries C, but fails. Begin rollback!
Trx 1: Compensating Action for B
Trx 2: Does B: ...
Trx 1: Compensating Action for A... does it set $10 to -$10? Trx1 Saga is "undone"!
Trx 2: Does C, the Trx2 Saga is complete!
```

A: This is a ==Saga [[Isolation]] Problem!== A Saga gives you atomicity, but it does not give you [[ACID]]-style [[Serializable Isolation|Serializable]] Isolation. Compensation is not a real rollback, it's more business operations happening later.
- General answer: ==Sagas do not provide isolation; you have to design the isolation strategy explicitly.== There is no universal automatic fix. 
- Standard approaches:
	- Define the real invariant first: "What must never be temporarily false?"
		- Two users can't hole the same unique resource
		- Inventory can't be oversold
		- A shipment can't be created for a cancelled order
		- A credit cannot be spent until finalized
	- Use state machines, not raw final mutations
		- Saga steps should usually create explicit states (our bug above comes from treating a pending state as if it were confirmed):
			- pending
			- reserved
			- authorized
			- confirmed
			- canceled
			- released
			- failed
	- Serialize by business key when needed
		- If two sagas touch the same important entity, run them through a per-key queue/actor lock.
		- For example:
			- All workflows for `order_123` go through one ordered stream
			- All workflows for `user_456` balance go through one writer
			- All workflows for `sku_789` inventory go through one reservation service
	- Use optimistic concurrency:
		- Each step writes with a precondition of an assumed version, and if the row changed underneath you, the saga does not blindly continues; it retries, branches, or compensates.
	- Make actions idempotent and compensations precise
		- A compensation doesn't mean: "Undo whatever the current state is", it should mean: Reverse the exact effect created by `saga_id=abc`.
	- Track dependencies when you allow them
		- If Saga B is allowed to depend on Saga A's not-yet-final result, that dependency must be made explicit, so that if A fails, B must be blocked/retried/compensated/manually reconciled. 
		- Implicit dependencies is where the nasty edge cases come from.

==Short answer==: Handle saga interleavings by combining business state machines, per-entity serialization where needed, optimistic preconditions, precise compensation, and reconciliation. If the invariant is too important to be temporarily violated, don't use a loose saga boundary for it.


Note that for things like money, you usually avoid mutable balances as the source of truth. Typically you'd model ledger entires like:
```
Initial posted balance: 40
Trx1:
	pending credit: 20
Trx2:
	tries debit -50


```
- ...then Trx2 should check against the posted/available balance, not raw balance including unresolved pending credits.
- If pending credits are spendable, then you need stronger semantics: Trx2 becomes dependent on Trx1, and if Trx1 fails, Trx2 must also be compensated, blocked, or converted into some debt/negative-balance state. That dependency must be explicit.

Bad saga step:
    Payment service increments balance from 40 to 60

  Better saga step:
    Payment service records pending_credit +20 for saga_123

