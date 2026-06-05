---
aliases:
  - Idempotent
  - Idempotency Key
---
An operation is Idempotent when ==performing the same operation multiple times has the same intended effect as performing it once.==

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


### Why does Idempotency matter?
Distributed systems are full of uncertainty! A client often can't distinguish between:
1. The server never received the request
2. The server received the request and crashed before acting on it.
3. The server received and processed the request, but crashed before returning a response.
4. The server is still currently processing it
5. The client timed out too aggressively

So... clients retry. Without idempotency, this can produce duplicate side effects (two charges, two orders, same event applied twice). 

Idempotency gives us the ability to ==safely== use [[At Least Once|At Least Once Delivery]], which is useful for real systems: "Try until it succeeds" is much easier than trying to guarantee [[Exactly Once|Exactly Once Delivery]].

> "Idempotency is how systems survive retries without pretending that retries won't happen."


