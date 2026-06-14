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

==NOTE:== The Transaction Outbox pattern guarantees that a committed business change will eventually have a durable corresponding message published. It *does not* guarantee that downstream consumers will process the message exactly once. The message relay can publish a message successfully and then crash before marking the outbox row as `published`, and the message gets published again by another relay worker. Because the usual delivery guarantee is [[At Least Once|At-Least-Once]] publication, consumers should be [[Idempotency|Idempotent]].

# Example

Creating the tables
```sql
CREATE TABLE orders (
  id UUID PRIMARY KEY,
  customer_id UUID NOT NULL,
  status TEXT NOT NULL,
  total_cents INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE outbox_messages (
  id UUID PRIMARY KEY,

  -- What domain object produced this message?
  aggregate_type TEXT NOT NULL,       -- e.g. 'Order'
  aggregate_id UUID NOT NULL,         -- e.g. orders.id

  -- What should be published?
  event_type TEXT NOT NULL,           -- e.g. 'OrderCreated'
  topic TEXT NOT NULL,                -- e.g. 'orders.events'
  payload JSONB NOT NULL, -- The actual message body to publish
  headers JSONB NOT NULL DEFAULT '{}', -- Optional message metadata (e.g. schema version)

  -- Relay bookkeeping
  status TEXT NOT NULL DEFAULT 'pending', -- e.g. pending/publishing/published/failed
  attempts INTEGER NOT NULL DEFAULT 0, -- Number of times the relay has tried to publish
  available_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  locked_at TIMESTAMPTZ, -- Time when relay worker claimed this message for publishing
  lock_expires_at TIMESTAMPTZ, -- Time when the lock expires
  locked_by TEXT, -- Identifier of relay worker that claimed this message

  published_at TIMESTAMPTZ, -- Time message was successfully published to broker
  broker_message_id TEXT, -- Optional ID returned by broker after successful publish
  last_error TEXT, -- Most recent publish error, if any

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Helps relay quickly find the messages that are readly to publish.
CREATE INDEX outbox_pending_idx
ON outbox_messages (available_at, created_at)
WHERE status = 'pending';

-- And those that are timed out
CREATE INDEX outbox_reclaim_expired_idx
ON outbox_messages (lock_expires_at, created_at)
WHERE status = 'publishing';
```

When the application creates an order:
```sql
BEGIN;

INSERT INTO orders (
  id,
  customer_id,
  status,
  total_cents
)
VALUES (
  'ord_11111111-1111-1111-1111-111111111111',
  'cus_22222222-2222-2222-2222-222222222222',
  'created',
  4999
);

INSERT INTO outbox_messages (
  id,
  aggregate_type,
  aggregate_id,
  event_type,
  topic,
  payload,
  headers
)
VALUES (
  'msg_33333333-3333-3333-3333-333333333333',
  'Order',
  'ord_11111111-1111-1111-1111-111111111111',
  'OrderCreated',
  'orders.events',
  '{
    "orderId": "ord_11111111-1111-1111-1111-111111111111",
    "customerId": "cus_22222222-2222-2222-2222-222222222222",
    "totalCents": 4999,
    "status": "created"
  }',
  '{
    "contentType": "application/json",
    "schemaVersion": "1"
  }'
);

COMMIT;
```

When a message relay polls, trying to claim work:
```sql
WITH claimable AS (
  SELECT id -- Select only the message IDs to claim.
  FROM outbox_messages -- Read from the outbox table.
  WHERE ( -- Start the condition for never-started messages.
      status = 'pending' -- Message has not been claimed yet.
      AND available_at <= now() -- Message is scheduled to be published now or earlier.
    )
    OR ( -- Start the condition for abandoned in-progress messages.
      status = 'publishing' -- Message was claimed by a relay worker.
      AND lock_expires_at <= now() -- Previous relay worker's lease has expired.
    )
  ORDER BY created_at -- Prefer older messages first.
  LIMIT 100 -- Claim at most 100 messages in this batch.
  FOR UPDATE SKIP LOCKED -- Lock these rows and skip rows another relay already locked.
)
UPDATE outbox_messages AS o -- Update the outbox rows selected above.
SET
  status = 'publishing', -- Mark each row as currently being published.
  locked_at = now(), -- Record when this relay acquired the lease.
  lock_expires_at = now() + interval '5 minutes', -- Allow another relay to reclaim after 5 minutes.
  locked_by = 'relay-worker-7', -- Record which relay worker owns the lease.
  attempts = attempts + 1 -- Increment the publish attempt count.
FROM claimable -- Use the claimed IDs as the update source.
WHERE o.id = claimable.id -- Update only rows selected by the claim query.
RETURNING o.*; -- Return claimed rows so the relay can publish their payloads.
```

And then for each returned outbox message, the relay might publish an event like:
```json
{
  "topic": "orders.events",
  "key": "ord_11111111-1111-1111-1111-111111111111",
  "type": "OrderCreated",
  "payload": {
    "orderId": "ord_11111111-1111-1111-1111-111111111111",
    "customerId": "cus_22222222-2222-2222-2222-222222222222",
    "totalCents": 4999,
    "status": "created"
  }
}
```
Important point: The relay can still publish duplicates. If it crashes before updating `outbox_messages.status = 'published'` (which would happen after this), then another relay instance later might see these rows as unpublished.

```sql
-- After successful broker publish:
UPDATE outbox_messages
SET
  status = 'published',
  published_at = now(),
  broker_message_id = 'orders.events:3:982734',
  locked_at = NULL,
  lock_expires_at = NULL,
  locked_by = NULL,
  last_error = NULL
WHERE id = 'msg_33333333-3333-3333-3333-333333333333';
```

On the consumer side, we want to make sure that we're [[Idempotency|Idempotently]] processing messages. If we're the only consumer of these messages, it might look something as simple as:

```sql
CREATE TABLE processed_messages (
  message_id UUID PRIMARY KEY,
  processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

And then the consumer does:
```sql
BEGIN;

WITH claimed AS (
  INSERT INTO processed_messages (message_id)
  VALUES ('msg_33333333-3333-3333-3333-333333333333')
  ON CONFLICT DO NOTHING
  RETURNING message_id
)
INSERT INTO order_notifications (
  order_id,
  customer_id,
  notification_type
)
SELECT
  'ord_11111111-1111-1111-1111-111111111111',
  'cus_22222222-2222-2222-2222-222222222222',
  'order_created'
WHERE EXISTS (SELECT 1 FROM claimed);


COMMIT;
```
Logic being:
- Try to insert `message_id` into `processed_messages`.
- If the insert succeeds, this consumer has not sprocessed the message before.
- If the insert conflicts, this consumer already processed the message before.
- Only run the consumer-side business change when the idempotency insert succeeded.

# Comparison with [[Change Data Capture]] (CDC)
- CDC typically trails the Database's [[Write-Ahead Log]] (WAL)/binlog and filters for inserts to (e.g.) the `ORDERS` table, and then updates the search index. The CDC connector keeps a long-lived connection to this log stream and reads new committed records as they appear, so it's typically pretty low latency (though you can also set your polling short for an Outbox+Message Relay version)
- Comparison
	- Transactional Outbox means the app explicitly writes an event as part of the same DB transaction as the business change. Important property is semantic intent: "OrderPlaced" or "InvoicePaid" exists as a durable event, because the app chose to create it.
	- In CDC, an external process watches the DB commit log and reacts to row changes. The important property is observation: "this row in `orders` changed was noticed after commit."
	- ==So the outbox is better when downstream systems need stable domain events. CDC is better when downstream systems just need a projection of database state, as is the case for search indexes, analytics, caches, or replicas.==