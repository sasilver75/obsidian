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



_________________

https://youtu.be/DOFflggE_0Q?si=LEAHey50kYougn51


You're building an application, and Transactions start easy: When a customer places an order:
1. Charge card
2. Reserve inventory
3. Create a ledger entry for accounting
You don't even think about this too much, your Database gives you ACID guarantees.
- Atomicity Either all three happen together, or none of them do!
- Isolation: While that transaction is in progress, no other part of your system can see the half-finished state. Another query checking a customer's balance won't see a charge for an order that hasn't fully processed yet.

Eventually, a single database begins growing, you get more data, more reads, more writes, and your database needs to be split up:
- Maybe you [[Sharding|Shard]]
- Maybe you break up your monolith into [[Microservice]]s where each service owns their own database.

Now your data lives on multiple independent machines instead of one.
- The payment flow that used to be one transaction against one database is now three completely separate operations against three separate database systems! (Payment Service, Inventory Service, Accounting Service)
- ==You can't wrap a transaction across these databases, because they don't know about eachother!==
	- If the payment works but the item is then out of stock, you can't just do a database rollback on the already-committed payment.
	- Partial failures like these aren't edge cases, they become routine at scale.

This calls for [[Distributed Transaction]]s, where you have a single logical operation that needs to span multiple independent databases or services, where all steps need to succeed together or be cleaned up when something goes wrong.


Textbooks give us two approaches:
- [[Two-Phase Commit]]
- [[Saga]] pattern

In practice, industry has chosen one over the other: Sagas.


In [[Two-Phase Commit]] (2PC):
![[Pasted image 20260606171704.png]]
We introduce a coordinator whose job is to make sure that all participants in a transaction agree on the outcome before any make their changes permanent:
1. Prepare Phase: Coordinator sends a message to each participant, asking "Can you commit this transaction?" Each database does the work, processes the request, durably records the change so that nothing is lost if it crashes, and locks effected rows, then responds "Yes,  I'm ready to commit", or "No, I can't do this."
	1. If anyone replies "No," the coordinator tells everyone to "abort" and roll back their changes.
2. Commit Phase:  If only "Yes"s are received, the coordinator tells each participant to make its change permanent (commit) and release their logs.
The Transaction is now complete!

==Fundamental problem==; ==It's a blocking protocol, which is dangerous, because it depends on multiple machines staying healthy at the same time.==
- Imagine that the coordinator receives all "Yes," but then crashes.
- The participants are now stuck, waiting for the coordinator to tell them to either "abort" or "commit".
- They just wait, and every other transaction in your system that needs to touch those rows is also blocked, waiting for locks to release that nobody will release.

![[Pasted image 20260606180744.png]]
==The other problem:== Tail Latency
- A single slow participant holds up the entire transaction! This is a [[Tail Latency]] problem.
- The Payments and Inventory service have to move at the speed of the slowest participant (e.g. holding a lock the entire time), and if a network partition means that people can't communicate... that further gunks things up

These 
Pat Helland paper: "Life Beyond Distributed Transactions"
- "Distributed transactions across autonomous services don't work at internet scale."

2PC does exist in production, but only inside distributed databases like [[Google Spanner|Spanner]], where the coordinator and participants are tightly coupled in the same system.


So what should we use instead? Enter [[Saga]] pattern.


When companies need to coordinate work across services, the Saga pattern is what they reach for. Uber, Netflix, Doormdash. Amazon all do this.

Saga starts from a ==different assumption==: "You don't need to have quick all-or-nothing atomicity spanning multiple services, you just need a way to eventually get to a consistent state, even when things go wrong along the way."

Instead of coordinating one big distributed transaction with locks held across services, you break the work into a chain of independent, local transactions, so each service does its piece of work, and commits to its own database on its own terms.
![[Pasted image 20260606181408.png]]
When something fails down the chain, there's no way to *roll back* to earlier steps, because things have already been committed upstream.
So you do what's called a [[Compensating Action]], which are business-level undos that reverse the effects of what already happened. A "refund" instead of a rollback. A "cancellation" instead of a an abort. How this works for *your system* is a key design consideration.


[[Saga|Sagas]] give you eventual consistency: A customer might briefly see a charge on their card before a refund goes through, but it always converges to a correct state, and nothing is blocked while that convergence is happening.

There are two ways to implement Sagas, and the choice between them determines who is responsible for detecting failures and running compensations.

1. Choreography
2. Orchestration

Choreography is the decentralized option, using a [[Publish-Subscribe|Pub Sub]] pattern where each service broadcasts an event when it finishes it work, and any interested service can pick it up if it would like to.

Payment service charges card and emits `CardCharged`
The Inventory listens for that, and then reserves stock and publishes an `InventoryReserved` event
The accounting service listens for this and records an entry.

A failing service publishes a failing event (e.g. `InventoryFailed`) and *upstream* services run compensating actions, publishing events as they go.

![[Pasted image 20260606181944.png]]

This works okay for a few services, but when you get a bunch of services...
![[Pasted image 20260606182004.png]]
Figuring out the current state of a given transaction becomes more difficult. Without a central place tracking all of this, you have to dig through logs across a dozen services trying to piece together what happened.


![[Pasted image 20260606182553.png]]
The second approach is ==Orchestration==, which is what most teams use.
- A central coordinator tells each service what to do, one step at at time.
	- "Card service, charge the card... waits for confirmation"
	- "Inventory service, reserve the stock... waits for confirmation"
	- If something goes wrong, the orchestrator knows how to tell pepole to run the right compensating transactions in the first order.


Tools like [[Temporal]] and [[Amazon Step Functions|AWS Step Functions]] are built to handle exactly this kind of orchestration.
- It's what HelloInterview uses to coordinate their payment flows, etc.

What happens when the Orchestrator crashes?
- It doesn't leave locks dangling across your system, it's durable.
- It's durable; when it starts back up, it knows where it was in a workflow and can pick back up where it started.

While [[Saga|Sagas]] fix the blocking problem of [[Two-Phase Commit|2PC]], they introduce  [[Compensating Action]]s, which themselves can get messy and bring their own type of complexity!
- "Just undo the previous step" sounds clean, but can get messy quickly!

Sounds clean, but gets messy
Example:
- Say the card charge went through and committed
- and then the inventory reservation failsbecause the item is out of stock.
- The compensation is  to issue a refund on the card, but that refund is visible to the customer; they see an actual charge show up, and then a few seconds later, they see a refund. Their bank might even send them a notification for each charge!
- It works, but It's not  a clean solution like a transaction rollback.

Example: Some are hard to undo at all!
- If a step in your flow is to send a confirmation email, then you can't unsend that email! 
- If you fired a webhook to a third party payment, you can send up a followup cancellation, but you can't guarantee that they'll see it in time.


On top of that ==compensating actions can themselves fail!==
- What if the Refund API is down when you need to issue that refund?
- Now you need retry logic for your compensation! And if you're retrying a refund, you need your action to be [[Idempotency|Idempotent]]!
- You end up needing the same level of reliability engineering for your failure handling as you do for your happy path.


Another failure mode:
- When your Payment service charges a card, let's say it needs to do two things:
	- Save the result to its own database
	- Publish an event to a Message Broker so that the next service in the chain knows that it's time to proceed (I guess we're talking Choreography). This is a [[Dual Write Problem|Dual Write]] problem.
![[Pasted image 20260606183815.png]]
- If the database write succeeds but the message publishing fails, then the next step in your Saga (Choreography) never gets triggered and the whole flow stalles.
![[Pasted image 20260606184120.png]]
- Conversely, you can have the opposite problem.

You can fix this with the [[Transactional Outbox Pattern]]
- Write both our data and the outgoing event into the same databae via  single local transaction:
	- Both into the `Payment` table and the `Outbox` (eg) table in a single transaction.
	- A separate background process watches the outbox table and publishes those events to a message broker.
		- Can use [[Change Data Capture]], tailing the database's own transaction log to pick up new entries.
		- Can otherwise simply poll the outbox table at a regular interval.


Decision Framework
- Avoid it if you can: ==If you can design your service boundaries so that data that transacts together lives in the same databases, do that!==
	- For example, move the inventory and ledger table into the same database that has the payments table if they always need to be update together anyways.
	- This way, a local database transaction is simpler, faster, and more reliable than any distributed alternative
- If you genuinely can't avoid it, you're going to use a [[Saga]], this genuinely is not a debate in the industry anymore.
	- Choreography can work for small-N number of services involved (e.g. an ECommerce notification system where an order is placed, an event triggers an email or push independently is a natural fit.) Don't over engineer from the get-go!
	- For anything more complex, Orchestration is the way to go. Most teams end up here, and tools like [[Temporal]] or [[Amazon Step Functions|AWS Step Functions]] make it practical to implement.


The pattern you'll see at most companies operating at scale:
- [[Saga]] with ==Orchestration==
- [[Idempotency|Idempotent]] Operations
- [[Transactional Outbox Pattern]]








