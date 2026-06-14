---
aliases:
  - DLQ
  - DL Queue
---
A separate [[Message Queue]] or store that holds messages that a system could not successfully deliver or process after the system's configured failure policy is exhausted.

Core Idea
> "Don't let one bad message keep blocking the normal message flow, but also do not silently discard the bad message."

A message is usually directed to a dead letter queue when there are:
- Too many processing failures (e.g. consumers repeatedly try to process the messages and fail)
- Explicit rejection by consumer (consumer says that the message is invalid or unprocessable)
- Queue policy violation (the message exceed limits such as size/schema/permissions)

[[Poisoned Message]]: A message that repeatedly causes processing failure because something about the message is bad for the current consumer: malformed JSON, missing required fields, invalid business state, unsupported scheme version, duplicate operations, and so on.

Dead Letter Queues usually do not "handle" the bad message by itself: It's mostly a quarantine area. The actual handling often requires manual inspection, repair logic, etc.

An original message like:
```json
{
  "order_id": "ord_123",
  "customer_id": null,
  "amount_cents": 4999
}
```
***May*** make its way to the dead letter queue as (see common designs below):
```json
{
  "original_queue": "orders", %% Might not be present depending on DLQ design %%
  "failure_reason": "customer_id is required",
  "attempt_count": 5,
  "failed_at": "2026-06-14T12:00:00Z",
  "original_message": {
    "order_id": "ord_123",
    "customer_id": null,
    "amount_cents": 4999
  }
}
```


Common designs:
1. One DLQ per main queue (`orders` has `orders-dlq`, etc.): Common in most systems.
	- Often cleaner because each DLQ has the same schema expectations, ownership, permissions, retention period, alerting policy, etc. as its source queue.
2. One shared DLQ for several queues (`ordres`, `payments`, ... all send failures to `shared-dlq`): Common for smaller systems, or when you want centralized triage.
3. One DLQ per subscription/consumer (The same published event may dead-letter separately for each subscriber): Common for publish/subscribe systems.
	- This is the [[Kafka]]-shaped version of the idea, typically with one DLQ per consumer group.
