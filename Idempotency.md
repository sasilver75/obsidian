---
aliases:
  - Idempotent
  - Idempotency Key
  - Idempotently
---
An operation is Idempotent when ==performing the same operation multiple times has the same intended effect as performing it once.==
- Idempotency is NOT [[Exactly Once]]; it means that repeated attempts produce one logical effect.
- Idempotency is NOT deduplication ("I saw this message before" is *one* way to implement idempotency)
- 

Mathematically:
```
f(f(x)) = f(x)
```

In System Desing terms: 
> If a request, message, job, or command is retried or delivered twice, the system does not accidentally duplicate teh side effect.


Example: 
```
POST /payments
Idempotency-Key: pay_abc123
```
If the client times out and retries with the same key, the server should not charge the card twice, for instance! IT should recognize that this logical operation was already processed, and ideally return the original result.


# Why does Idempotency matter?
Distributed systems are full of uncertainty! A client often can't distinguish between:
1. The server never received the request
2. The server received the request and crashed before acting on it.
3. The server received and processed the request, but crashed before returning a response.
4. The server is still currently processing it
5. The client timed out too aggressively

So... clients retry. Without idempotency, this can produce duplicate side effects (two charges, two orders, same event applied twice). 

Idempotency gives us the ability to ==safely== use [[At Least Once|At Least Once Delivery]], which is useful for real systems: "Try until it succeeds" is much easier than trying to guarantee [[Exactly Once|Exactly Once Delivery]].

> "Idempotency is how systems survive retries without pretending that retries won't happen."


# Common Use Cases
- Payments: A payment API should not double-charge because the client retried after a timeout.
- Order Creation: A user double-clicks "Place Order," or the browser retries. You should create one order.
- Inventory Reservation: A saga may reserve inventory before payment; retrying "reserve inventory" shouldn't reserve twice.
- Webhook Handling: Providers often deliver webhooks more than once. Your system should avoid applying it 2x.
- Background jobs: A worker might process a job and then crash before acking it, then the queue redelivers it. The job handler should be idempotent (checking if email has already been sent, or using a stable email/send record).
- Sagas: Each saga step *and* compensating action should be idempotent, or you'll have serious reliability problems.
- Email and Notifications: Can be tricky, because external side effects might not be reversible. You can make your system idempotent by recording: notification_id=password_reset:user_123:token_456, but true e2e idempotency depends on the provider too - if they aren't idempotent...


# Design Checklist
1. What's hte logical operation eidentity?
2. Who generates the idempotency key?
3. Where is the key stored?
4. How long is the key retrained?
5. What happens if the same key arrives with different parameters?
6. What happens if the original request is still processing?
7. What result should a retry receive?
8. Are all side effects covred, including emails/queues/webhooks/external APIs?
9. Is there a unique constrain or atomic write protectingraces?
10. Can failed/partial operations be retried safely?

The strongest implementations usually combine:
1. A stable operation ID
2. A database uniqueness constraint
3. A stored request hash
4. A stored final response/result
5. retry-safe side effects


### Naturally Idempotent Operations
Some operations are naturally idempotent by their shape:
```sql
UPDATE users
SET email_verified = true
WHERE id=123;
```
Doing it once or 10 times has the same result.

Other operations are *not* idempotent
```sql
UPDATE accounts
SET balance = balance - 100
WHERE id = 123;
```
Two repeated calls would deduct $200; not what our users likely want!


### HTTP Examples
- HTTP has conventional idempotency semantics:
	- `GET`: Should be safe ad idempotent, without side-effects.
	- `PUT`: Usually idempotent: replace resource with this exact representation
	- `DELETE`: Usually idempotent: Ensure resource is deleted
	- `POST`: Not idempotent by default

A non-idempotent `POST`:
```
POST /orders
{
	"itemId": "book_1"
}
```
Calling this multiple times would create multiple orders.

But if we use an [[Idempotency|Idempotency Key]] with a header:
```
POST /orders
Idempotency-Key: order_attempt_789
{
	"itemId": "book_1"
}
```
In this way, we can recognize subsequently-received requests as still being the same logical intent.

# How to achieve Idempotency

### 1) Use Natural State Transitions
- Prefer "set state to X" over "do incremental mutation"

Instead of
```sql
UPDATE inventory
SET reserved_count = reserved_count + 1
WHERE sku = 'abc';
```
instead do
```sql
INSERT INTO reservations (reservation_id, sku, quantity)
VALUES ('res_123', 'abc', 1)
ON CONFLICT (reservation_id) DO NOTHING;
```
In this case, repeating the same command doesn't create another reservation.


### 2) Use Idempotency Keys
- The client generates a unique key for a logical operation.
- Servers stores the key with the result:
```
  idempotency_key | request_hash | status      | response
  ------------------------------------------------------------
  abc123          | h1           | completed   | payment_999
```
Above: We store request hash so that the cilent can't use the same idempotency key with a different request body!
- On the first request:
	- Server receives key `abc123`
	- Checks whether key exists, it does not, so it processes the request
	- Stores the outcome before returning the response.
- On retry:
	- Server receives key `abc123` again, finds the stored result, and returns the same result instead of repeating the side effect


### 3) Uniqueness Constraints
- Databases are good at enforcing idempotency boundaries!
```sql
CREATE UNIQUE INDEX unique_payment_attempt
ON payments(idempotency_key);
```
Then:
```sql
INSERT INTO payments (idempotency_key, user_id, amount)
VALUES ('pay_123', 42, 5000)
ON CONFLICT (idempotency_key) DO NOTHING;
```
This prevents duplicate rows even under concurrent retries.
These are one of the most reliable idempotency tools because they protect you even if two requests race.


### 4) Store Request Hashes
- If a client reuses the same idempotency key with a different request body, that's dangerous!

```
Key: abc123 
First request: charge $10
Second request: charge $500
```
The server should reject this, so systems often store a hash of the original request parameters. If the same key appears with a different hash, return an error like: `409 Conflict: idempotency key reused with different parameters`.


### 5) Track In-Progress Requests
- Concurrent duplicate requests can arrive at the same time.
```
Requset A with key abc123 starts processing
Request B with key abc123 arrives before A finishes
```
The server needs a policy!
We need to record "someone is already processing this key" before doing the actual dangerous work!
So when this request arrives:
```
POST /payments
Idempotency-Key: abc123
...
```
The server keeps an idempotency table:
```
key     status       response
abc123  processing   null
```

Flow:
1. Request A arrives with key `abc123`
2. Server atomically inserts `abc123` with a status `processing`
3. Request A starts charging the card
4. Request B arrives with the same key before A finishes.
5. Server sees `abc123` is already `processing`
6. Server does **not** charge the card again.

For request B, we typically return "still processing" (either `409` or `202`)


### 6) Idempotent consumers
- In a message-driven systems, consumers should assume taht messages may be delivered more than once:
```
{
	"event_id": "evt_123",
	"type": "PaymentCaptured",
	"payment_id": "pay_456"
}
```
The consumer stores processed event IDs
```
INSERT INTO processed_events (event_id)
VALUES ('evt_123')
ON CONFLICT DO NOTHING
RETURNING event_id;
```
If the insert succeeds (returning a row), process the event. If it conflicts, skip it.
This is common with Kafka/SQS/RabbitMQ, etc.


### 7) Inbox and Outbox Patterns
- The inbox pattern deduplicates incoming messages:
```
incoming event -> inbox table -> process once
```
- The outbox pattern makes outgoing messages reliable by storing them in the same local transaction as the business change.
```
update order AND insert OrderCreated event into outbox table in ONE transaction
Commit
Message relay/publisher later reads this table and sends event to a queue
```


### Stable Resource IDs
- Another simple technique: Let the client provide the ID of the thing being created.
Instead of  `POST /orders`, what if you did `POST /orders/order_abc123`, so that repeated requests target the same resource.
- This is good/fine when the operation naturally creates a named resource.
