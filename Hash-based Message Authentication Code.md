---
aliases:
  - HMAC
---
A specific kind of [[Message Authentication Code]] that uses a secret key, a [[Hash|Cryptographic Hash Function]], and a message to produce an [[Authentication Tag]] that lets a receiver check message integrity and shared-secret authenticity.

This is the same abstract shape as a MAC
```
Authentication Tag = HMAC(secret_key, message)
```
HMAC has the same external shape, but internally uses a hash function in a carefully structured two-layer construction.

The receiver recomputes the HMAC over the received message using the same secret key. If the recomputed Authentication Tag matches the received Authentication Tag, the receiver accepts the message.

> A tamper-evident seal made by combining a message with a shared secret in a carefully-designed way before hashing.

It's not simply `Hash(secret_key || message)`.
That naive construction can be vulnerable with many hash function (e.g. length extension attacks). 
HMAC instead hashes the key and message in two layers:
```
HMAC(K, m) =
  H((K' xor opad) || H((K' xor ipad) || m))
```
Where
- `H` is the hash function (e.g. [[SHA-256]])
- `K` is the secret key
- `K'` is the key normalized to the hash function's block size
- `m` is the message
- `ipad` is an inner padding constant
- `opad` is an outer padding constant
- `||` means byte concatenation

# Use Cases

### [[Webhook]] verification
- Suppose Stripe/Github sends your server a webhook payload and an HMAC Authentication Tag.
- Your server recomputes the HMAC over the exact raw request body using a shared secret.
- If the Authentication Tag matches, your server knows the payload was not modified and was produced by someone who knows the webhook secret.

API request authentication
Suppose a client wants to call a private API endpoint:
```http
POST /v1/payments
Host: api.example.com
Date: Fri, 12 Jun 2026 18:00:00 GMT
X-Client-Id: client_123
X-Signature: hmac-sha256=...
```
The client computes an [[Hash-based Message Authentication Code|HMAC]] over selected parts of the request:
```
message_to_authenticate =
  HTTP method + path + timestamp + request body

Authentication Tag =
  HMAC-SHA-256(api_secret_key, message_to_authenticate)
```
Because the server has the same `api_secret_key` ([[Symmetric Key Encryption|Symmetric Encryption]]), when the request arrives, the server recomputes the HMAC over the same method, path, timestamp, and request body. If the Authentication Tag matches, the server knows:
1. That the request wasn't modified in transit
2. That the request came from someone who knows the API secret key
3. That the authenticated fields, such as the path and body, are bound together


### [[Cookie]] verification
- Suppose a server wants to store a small amount of arbitrary session state in a browser cookie:
```
user_id=123&role=user&expires=2026-06-12T20:00:00Z
```
(Don't worry about whether this stored state is realistic)

- If the server sends that cookie without protection, the user could always edit it, or manually send a new cookie. The point of the cookie is "This is state the server is storing on the client," but the client ultimately (even in the case of an HttpOnly cookie) can control what data is sent on requests to the server. They may send:
```
user_id=123&role=admin&expires=2026-06-12T20:00:00Z
```
Oops, the user has seemingly escalated their privileges! (Or done something else in a diff. situation)

So the server attaches an [[Hash-based Message Authentication Code|HMAC]] [[Authentication Tag]]:
```
cookie_value =
  base64(user_id=123&role=user&expires=2026-06-12T20:00:00Z)
  .
  HMAC-SHA-256(server_secret_key, encoded_payload)
```
Now, when the browser sends the cookie back, the server recomputes the HMAC over the encoded payload. 
- If the Authentication Tag matches, the server accepts that the cookie payload was produced by the server and not modified by the browser.

This is common in systems where the server wants tamper detection without server-side storage for every session.
- Note that in this case the signed cookie is authenticated, not necessarily encrypted.
- The user can still read the cookie contents.
- The user cannot modify the cookie contents without invalidating the HMAC Authentication Tag.


