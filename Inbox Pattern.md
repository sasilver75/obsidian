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


==FIX==
```sql
BEGIN;

INSERT INTO processed_messages (consumer_name, message_id)
VALUES ('billing-worker', 'event_123')
ON CONFLICT DO NOTHING
RETURNING message_id;

-- Application branch:
--   If zero rows returned:
--     COMMIT;
--     ACK/drop this duplicate delivery;
--     STOP before business logic.
--
--   If one row returned:
--     this worker owns the message and may run business logic.
UPDATE ...
INSERT ...

COMMIT;

-- ACK the broker message only after COMMIT succeeds.
```
In a Postgres-style system, the second worker would usually block or get the conflict at the insert into the inbox table, not after doing all the downstream work. The uniqueness constraint is the admission gate.

The inbox pattern does not buy "the work is naturally idempotent." It buys "idempotency by deduplication." In the Inbox pattern, only the first delivery is allowed past the gate. If the operation is already naturally idempotent and concurrency-safe, then yes, the inbox may not buy much. But many message handlers are not like that! The inbox is valuable when the duplicate would otherwise create a second business effect.


# Q: Is there a reason to have a status field? It seems to me like the presence of the inbox record in the table is enough, if the "business work" being done can be done in the same transaction.

A: You're right! ((Lol)).  If the inbox insert and the business effect can happen in the same database transaction, status is not required for the core deduplication effect.

In that strongest simple version, the invariant is:
```
If the inbox row exists, the business effect committed.
If the business effect did not commit, the inbox row does not exist.
```
In this world, something like this would totally work, with no `status` needed.
```sql
BEGIN;

INSERT INTO processed_messages (consumer_name, message_id)
VALUES ('ledger-service', 'event_123')
ON CONFLICT DO NOTHING
RETURNING message_id;

-- Application branch:
--   If zero rows returned:
--     COMMIT;
--     ACK/drop this duplicate delivery;
--     STOP before business logic.
--
--   If one row returned:
--     continue to business work.
INSERT INTO ledger_entries (...);

COMMIT;

-- ACK the broker message only after COMMIT succeeds.
```


Use `status` when the inbox table needs to represent message processing state, not just message completion existence. 

For example, when the work is too long-running to safely keep one database transaction open.

Example: An image-processing consumer.
A queue delivers:
```json
{
  "message_id": "img_evt_123",
  "image_id": "img_456"
}
```

The consumer needs to generate thumbnails, extract metadata, run moderation, and upload derived files. This might take 30 seconds or more! We typically don't want to hold open a database transaction for this long!

So in this case, an inbox row (with a status) will represent a claim:
```sql
CREATE TABLE inbox_messages (
  consumer_name text NOT NULL,
  message_id text NOT NULL,

  status text NOT NULL,
  locked_until timestamptz,
  locked_by text,

  attempt_count integer NOT NULL DEFAULT 0,
  next_attempt_at timestamptz,
  last_error text,

  received_at timestamptz NOT NULL DEFAULT now(),
  processed_at timestamptz,
  dead_lettered_at timestamptz,

  CHECK (status IN ('processing', 'processed', 'failed', 'dead_lettered')),

  PRIMARY KEY (consumer_name, message_id)
);
```

When a worker inserts:
```sql
BEGIN;

INSERT INTO inbox_messages (
  consumer_name,
  message_id,
  status,
  locked_until,
  locked_by,
  attempt_count
)
VALUES (
  'image-processor',
  'img_evt_123',
  'processing',
  now() + interval '60 seconds',
  'worker-b',
  1
)
ON CONFLICT DO NOTHING
RETURNING *;

-- Application branch:
--   If zero rows returned:
--     ROLLBACK or COMMIT this tiny transaction;
--     inspect/steal the existing row in a separate step.
--
--   If one row returned:
--     COMMIT this claim transaction;
--     process the image while periodically renewing the lease.

COMMIT;
```
You can see that we've included an expiration, so that a record here is basically more of a "claim".

If that returns a row:
```
Worker B created the inbox row.
Worker B owns the message.
Worker B processes the image.
```
If that returns no row, Worker B did not get the claim. 


If the worker crashes, a later worker needs to know:
1. Was `img_evt_123` completed?
2. Was `img_evt_123` claimed by a worker that seems to have perhaps died, and maybe or maybe didn't complete the work?


When processing of the actual work finishes:
```sql
-- While the worker is still processing, it periodically renews its lease.
-- If this returns zero rows, the worker no longer owns the message and
-- should stop before performing more non-idempotent work.
UPDATE inbox_messages
SET locked_until = now() + interval '60 seconds'
WHERE consumer_name = 'image-processor'
  AND message_id = 'img_evt_123'
  AND status = 'processing'
  AND locked_by = 'worker-b'
  AND locked_until > now()
RETURNING *;

-- When processing finishes, mark the row processed only if this worker
-- still owns the active lease.
UPDATE inbox_messages
SET status = 'processed',
    processed_at = now(),
    locked_until = NULL,
    locked_by = NULL
WHERE consumer_name = 'image-processor'
  AND message_id = 'img_evt_123'
  AND status = 'processing'
  AND locked_by = 'worker-b'
  AND locked_until > now()
RETURNING *;
```
Without `status`, the mere presence of the row would incorrectly look like "already processed," even though the worker may have died halfway through.


The flow for a worker B would typically be:
1. Try to insert/claim.
2. If insert succeeds, Worker B owns the work.
3. If insert conflicts, inspect or conditionally update the existing row.
4. Based on the existing row, skip, wait, or steal.

```sql
UPDATE inbox_messages
SET
  locked_by = 'worker-b',
  locked_until = now() + interval '60 seconds',
  attempt_count = attempt_count + 1
WHERE consumer_name = 'image-processor'
  AND message_id = 'img_evt_123'
  AND status = 'processing'
  AND locked_until <= now()
RETURNING *;
```
If this returns a row, then:
- The previous worker's lease expired.
- Worker B successfully stole/reclaimed the message.
- Worker B processes the image.

If that returns no row, Worker B checks why:
```sql
SELECT
  status,
  locked_until,
  locked_by,
  attempt_count,
  next_attempt_at,
  last_error
FROM inbox_messages
WHERE consumer_name = 'image-processor'
  AND message_id = 'img_evt_123';
```
What you do next depends on the status that you see:


| Step | Worker B action                                          | SQL shape                                                                                                       | If it succeeds                                                                                                                 | If it does not succeed                                                                                                                 |
| ---- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Try to create the inbox row                              | `INSERT ... ON CONFLICT DO NOTHING RETURNING *`                                                                 | Worker B made the first claim and may process the message.                                                                     | Another worker already created the row. Go to step 2.                                                                                  |
| 2    | Try to steal only if the existing claim is stale         | `UPDATE ... WHERE status = 'processing' AND locked_until <= now() RETURNING *`                                  | Worker B reclaimed an expired lease and may process the message.                                                               | The message is either already done, actively owned, failed, dead-lettered, or another worker beat Worker B to the steal. Go to step 3. |
| 3    | Inspect the existing row                                 | `SELECT status, locked_until, attempt_count FROM inbox_messages WHERE consumer_name = ... AND message_id = ...` | Worker B learns the current state.                                                                                             | If the row somehow does not exist, retry from step 1.                                                                                  |
| 4A   | Existing row is `processed`                              | No write needed                                                                                                 | Skip duplicate work. Usually acknowledge/drop this delivery.                                                                   | Not applicable.                                                                                                                        |
| 4B   | Existing row is `processing` and `locked_until > now()`  | No write needed                                                                                                 | Another worker currently owns the lease. Do not process. Release/nack/retry later, or let broker visibility timeout redeliver. | Not applicable.                                                                                                                        |
| 4C   | Existing row is `processing` and `locked_until <= now()` | Retry step 2                                                                                                    | The old lease is stale. Worker B may try the atomic steal again.                                                               | Another worker may steal first.                                                                                                        |
| 4D   | Existing row is `failed` and retry is allowed            | `UPDATE ... WHERE status = 'failed' AND next_attempt_at <= now() RETURNING *`                                   | Worker B claims the retry and processes the message.                                                                           | Retry is not due yet, retries are exhausted, or another worker claimed it.                                                             |
| 4E   | Existing row is `dead_lettered`                          | No write needed                                                                                                 | Skip processing and optionally alert or record duplicate delivery.                                                             | Not applicable.                                                                                                                        |

I imagine that the various statuses that we'd have would be:
- processing -> processed
- processing -> failed
- failed -> processing (for a retry, incrementing attempt_count) 
- failed -> dead_lettered (when we reach some maximum attempt_count)

****
