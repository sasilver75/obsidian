---
aliases:
  - Inbox
---


A reliability/[[Idempotency]] pattern for message consumers (context: [[Message Queue]])

A service stores incoming message IDs, or or the whole incoming message, in a durable inbox table before or while processing the message.
- ==This inbox must be durable and **shared*** by all instances of the same logical consumer service!==
- "The inbox pattern deduplicates message processing across all instances of the same logical consumer, as long as they share a durable inbox store with a uniqueness constraint!"

If the same message is delivered again, the service can detect that it has already handled it, and avoid repeating the same side effect.

Typical flow:
1. Consumer receives message `event_123`
2. Consumer inserts `event_123` into inbox table with a uniqueness constraint
3. If insert succeeds, process the message and update business state.
4. If insert conflicts, this message was already processed or is already being processed.

It's useful because queues/webhooks/brokers often provide [[At Least Once]] delivery, meaning duplicate messages can happen. This is a way of providing [[Idempotency]] by just not doing duplicate work via deduplication.

Pretty similar to the idea of an [[Idempotency|Idempotency Key]] Table, just applied at a different boundary (protecting message consumption, rather than an API command).

