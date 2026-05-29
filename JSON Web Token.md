---
aliases:
  - JWT
---
A [[Signing|Signed]] token that carries claims about an identity or request context, encoded with [[JSON]]

Useful because backend services can validate tokens locally without calling the auth server on every request. `Client -> API Gatewawy -> Service A -> Service B`. Each service *can* verify the token signature and claims independently.
- Note: This is one way it can work (JWT Pass-Through).
	- Alternatively, the [[API Gateway]] can verify and then inject trusted headers about identity/authorization. Critically, the services then must only trust those headers from the *gateway* or trusted proxy. If someone can call Service A directly and set `X-UserId: admin`, then the system is broken.
	- Alternatively, the Gateway can verify, and then mint a short-lived *internal token* (a [[JSON Web Token|JWT]]). 
	- Alternatively, sometimes Service A doesn't forward the user JWT to Service B. Instead, it calls B using its own service credentials, and optionally includes user context separately.
	- A simple rule: If downstream services make security-sensitive decisions, do not rely on unsigned headers unless the network/proxy boundary is extremely well controlled.

==A JWT has three parts==:
- Header
- Payload
- Signature

Conceptual Example:
```JSON
 header:
  {
    "alg": "RS256",
    "kid": "key-123",
    "typ": "JWT"
  }

  payload:
  {
    "iss": "https://auth.example.com",
    "sub": "user_42",
    "aud": "orders-api",
    "exp": 1760000000,
    "scope": "orders:read"
  }

  signature:
  signed(header + payload)
```
Important parts:
- `iss`: issuer, who issued the token
- `sub`: subject, who the token is about
- `aud`: audience, which service the token is intended for
- `exp`: expiration time
- `iat`: issued-at time
- `nbf`: not-before time
- `scope/roles`: permissions
- `kid`: key ID in the header, used to find the right verification key


Pros:
- Stateless validation, good for distributed systems. Standard format, works across languages/services. Supports key rotation with [[JSON Web Key Set|JWKS]]. Can carry user or service identities.
Cons:
- Hard to revoke before expiration. Easy to misuse by trusting claims without validation. Tokens can become too large. Stale permissions if roles are embedded. Leaked bearer token can be replayed until expiration. Requires careful key rotation and caching.

