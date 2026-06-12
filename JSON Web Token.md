---
aliases:
  - JWT
---
A [[Cryptographic Signature|Signed]] token that carries claims about an identity or request context, encoded with [[JSON]].
- It's important that the JWT is usually signed, not encrypted; anyone holding it can decode the claims. They just can't modify it without "breaking" the signature, when it's later validated.

Useful because backend services can validate tokens locally without calling the auth server on every request. `Client -> API Gatewawy -> Service A -> Service B`. Each service *can* verify the token signature and claims independently.
- Note: This is one way it can work (JWT Pass-Through).
	- Alternatively, the [[API Gateway]] can verify and then inject trusted headers about identity/authorization. Critically, the services then must only trust those headers from the *gateway* or trusted proxy. If someone can call Service A directly and set `X-UserId: admin`, then the system is broken.
	- Alternatively, the Gateway can verify, and then mint a short-lived *internal token* (a [[JSON Web Token|JWT]]). 
	- Alternatively, sometimes Service A doesn't forward the user JWT to Service B. Instead, it calls B using its own service credentials, and optionally includes user context separately.
	- A simple rule: If downstream services make security-sensitive decisions, do not rely on unsigned headers unless the network/proxy boundary is extremely well controlled.

==A JWT has three parts==:
- ==Header==: Metadata about the token/signature, like the algorithm used and the key id `kid`
- ==Payload==: The JSON body containing claims.
- ==Signature==: Cryptographic proof that the token was signed by the issuer and not modified.

Conceptual Example:
```JSON
 header:
  {
    "alg": "RS256",
    "kid": "key-123",
    "typ": "JWT"
  }

  payload:
  %% A "claim" is juat just a statement inside the token; these are all claims (iss, sub, ...) %%
  {
    "iss": "https://auth.example.com",
    "sub": "user_42",
    "aud": "orders-api",
    "exp": 1760000000,
    "email": "alice@example.com",
    "email_verified": true,
    "iat": 1759996400,
    
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


# How do we issue JWTs? ((Not sure I love this))
- Once we've authenticated a user , your auth server creates a JSON payload of claims and signs it with a key.
1. Alice submits email/password
2. Acme (our app) verifies password against its user DB
3. Acme decides Alice is local user_id=42
4. Acme creates JWT claims
5. Acme signs those claims with its private signing key
6. Acme returns the JWT to the client
7. Later, Acme APIs verify the JWT signature and read user_id=42

An example JWT header:
```json
{
	"alg": "RS256",
	"kid": "acme-key-2026-06",
	"typ": "JWT"
}
```

An example JWT payload might look like:
```json
{
    "iss": "https://auth.acme.com",
    "sub": "user_42",
    "aud": "https://api.acme.com",
    "iat": 1760000000,
    "exp": 1760000900,
    "scope": "projects:read projects:write"
}
```

So we then take our signing key, and sign:
`SIGN ( base64url(header) + "." + base64url(payload) )`

So our final token is then:
```
base64url(header).base64url(payload).base64url(signature)
```

So our login response to a user's login/password `POST` might be something like:
```json
{
    "access_token": "eyJhbGciOiJSUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 900
}
```

Honestly though, for many normal browser apps, the idea is that we don't want to make the browser hold long-lived, powerful JWTs unless you have a clear reason. It's often better to use opaque session [[Cookie]]s.

JWTs are still useful, but are more often for:
- API-to-API auth
- Mobile apps
- CLIs
- [[Single Page Application|SPA]]s with careful token handling
- Short-lived API access tokens
- Tokens exchanged between internal services
- Third-party API access


# What reads paths look like, with user-supplied JWTs?

Imagine you're a server, and you receive an incoming request:
```
GET /api/v1/projects/123
Authorization: Bearer eyJhbGci...
```
That [[JSON Web Token|JWT]] is not automatically trusted -- it's just a base-64 string that the client handed to us!

1. We extract the actual `eyJhbGci...` encoded token
2. We decode the header and payload.
```
// header
{
	"alg": "RS256",
	"kid": "google-key-1"
}

// payload / claims
{
	"iss": "https://accounts.google.com",
	"sub": "109837465",
	"aud": "api.acme.com",
	"exp": 1760000000,
	"scope": "projects:read"
}
```
...But at this point, these are just claims. The server must treat them as unverified until we can verify the signature!

3. Find the appropriate Public Key
The server looks at:
- `iss`: Who *supposedly* issued it
- `kid`: Which signing key was used
Then it fetches of retrieves cached keys from the issuer's [[JSON Web Key Set|JWKS]] endpoint.

4. Verify the Signature
The server checks:
- Was this JWT signed by the issuer's private key?
- Was the payload modified at all?
- Is the signing algorithm acceptable?
If the signature verification fails, reject!

5. Validate the Claims
The server then checks things like:
- `iss`: Is this the expected issuer?
- `aud`: Is this token meant for this API/app?
- `exp`: Not expired?
- `nbf`: Valid yet?
- `scope/permissions`: Enough for this action?
- `token type`: Access token vs id token? For an API, you usually want an access token, not an id token.

6. Identify the Caller
The server uses: `iss` + `sub` to identify the external user, or maps it to a local user.
`https://accounts.google.com + 109837465 -> acme_user_id=42`
If it's the case that it's a first-party JWT issued by *our own systems*, then the "sub" (subject) is typically just the user id.

7. Authorize the Action
Now, the server asks: Can `user 42` actually read `project 132`?
- It might use token scopes:
	- `projects:read`
- But also usually checks local `Acme` data: Is this specific user allowed to do this specific thing on this specific resource?
	- Both must pass:
		- ==Scope check:== Is this token allowed to perform this class of action?
			- Does the token have projects:read?
		- ==Local authz:== Is this principal allowed on this particular resource?
			- Is user_42 a member a project 123?

8. Allow or reject the action
- Return a `200 OK` or a `401 Unauthorized`/`403 Forbidden`



# It seems like [[Session]]s are preferred for first-party web applications. What should we use JWTs for?

Yes, opaque server-side sessions are often the simpler and safer default

You usually want a JWT when you specifically need ==portable, signed claims that another system can verify without calling your session database every time.==

Good use cases:
1. Short-lived API access tokens
	- A common pattern is `opaque refresh token / session` -> short-lived JWT access token`
	- The JWT lasts maybe 5-15 minutes. If it's leaked, damage is limited. Revocation happens by revoking the refresh token.
2. Microservices/distributed systems
	- Service A receives a request and passes identity to Service B. Service B can verify the JWT signature locally instead of calling the auth service on each request.
3. Third-party APIs
	- If external clients call your API, a JWT lets them present a signed token with scopes, tenant ID, expiry, issuer, etc.
4. Federated login/OIDC
	- [[OpenID Connect]] uses JWTs for ID tokens. If you integrate with Google/Auth0/Okta/Clerk, you'll encounter JWTs because they're a standard way to transmit signed identity claims.
5. Edge/serverless environments
	- IF your code runs across many regions/edge workers, verifying a signed token locally can be much faster and simpler than reaching a central control store.
6. Scoped delegation
	- JWTs are useful when you want a token that says:
```
user: 123
scope: read:invoices
tenant: acme
expires: 10 minutes from now
```
So that the receiver can verify (with a [[JSON Web Key Set|JWK]] that the token was issued by you and hasn't expired.)



# Where should we store JWTs?

It depends on the application shape!
- In a Web App with [[Backend for Frontend|BFF]] pattern
	- The browser talks only to your BFF, and the BFF talks to downstream APIs. In this setup, the browser doesn't *need* the JWT access token; the browser can just hold a normal session cookie. This is often the best web security posture. The long-lived credential stays server-side, protected from browser JavaScript. 
	- Access JWT storage: Do not store JWT in browser. 
	- Refresh Token storage: Server/BFF stores it, browser uses an `HttpOnly, Secure, SameSite` [[Cookie]] session token.
- In a [[Single Page Application|SPA]] that calls backend APIs directly
	- Here, the frontend JS is the API client, so the API expects an `Authorization: Bearer eyJ...`. So the JS must have access to the JWT somehow. The preferred pattern is keeping the access JWT in memory only, and the refresh token in a `HttpOnly Secure SameSite` Cookie. If the user refreshes  the page, the in-memory access token disappears, and the app calls `/refresh`, automatically sending the `HttpOnly` refresh token stored in cookies, and the server returns a fresh JWT access token.
	- Access JWT storage: Memory-only, if possible. 
	- Refresh token storage:  `HttpOnly Secure SameSite` [[Cookie]] via your backend/token endpoint
- In a pure browser [[Single Page Application|SPA]] with no backend
	- In this scenario, your pap is just static files plus browser JS. There isn't a trusted application server that can hold secrets or manage sessions. ==This is the hardest model to secure cleanly.==
	- Access JWT storage: Memory; [[Session Storage]] only as a tradeoff; avoid `localStorage`
	- Refresh token storage:  Avoid if possible; if required, rotate/reuse-detect and understand JS exposure
- In a native mobile application
	- Native app isn't a browser, it has access to OS-provided secure storage. The access token should still be short-lived; the refresh token is the valuable credential, so you store using the platform's protected storage. Mobile can store refresh tokens more reasonably than a pure browser SPA because the OS gives you a better place to put secrets.
	- Access JWT storage: Memory for access token.
	- Refresh token storage:  iOS Keychain/Android Keystore-backed secure storage
- In a CLI/Desktop application
	- A CLI or Desktop application is also a public client, but it might have access to a system keychain (e.g. macOS Keychain, Windows Credential Manager, Linux Secret Service/libsecret).
	- Access JWT storage: Memory for access token.
	- Refresh token storage:  OS keychain/keyring, or protected config file if unavoidable
- In a server-to-server context
	- Here, there's no human user session. You typically don't want a user refresh token at all, you want a machine credential. JWTs are often used here because services can validate them locally, but you still want short expirations, `aud` restrictions, issuer validation, and key rotation.
	- Access JWT storage: Memory/env/secret manager
	- Refresh token storage:  Usually no user refresh token; use client credentials or workload identity


The key point:
```
If JavaScript must attach `Authorization: Bearer <token>`, then JavaScript must be able to access the Access Token.
```
So keep that access token short-lived and in-memory; if you put it in [[Local Storage]], [[Cross-Site Scripting|XSS]] can steal it. If you put it in an `HttpOnly` cookie, JS can't attach it as a Bearer header; at that point, you're really doing cookie-based auth, often with a BFF pattern.




# How does JWT [[Cryptographic Signature|Sign]]ing work?
- (Note that Signing and [[Authentication Tag]]s are technically different, but people commonly use "signing" to refer to either of them. In the JWT case, I think it's common that you indeed use Private key to sign it, and then Public keys (in the form of [[JSON Web Key Set|JWKS]]) to validate it. So it indeed is better referred to as "signing.")
- The signer doesn't like "The JSON object" in some abstract sense, they sign this exact byte string:
```
base64url(UTF8(protected_header)) + "." + base64url(payload)
```
then the final compact token is:
```
base64url(header).base64url(payload).base64url(signature)
```

The signature only answers: "Were these exact header and payload bytes produced by someone with the right signing key, and will the bytes have changed since then?"

Header:
```json
{
  "alg": "RS256",
  "typ": "JWT",
  "kid": "key-2026-01"
}
```
Payload:
```json
{
  "iss": "https://auth.example.com",
  "sub": "user_123",
  "aud": "payments-api",
  "exp": 1781300000,
  "scope": "invoice:read"
}
```

The signer [[Base64]]url-encodes both pieces, joins them with a period, and then signs taht joined string.

==There are two main signing styles:==

| Style                           |                                                          Example `alg` | Key model               | Who can sign?                 | Who can verify?                                                 |
| ------------------------------- | ---------------------------------------------------------------------: | ----------------------- | ----------------------------- | --------------------------------------------------------------- |
| [[Message Authentication Code]] | `HS256` ([[Hash-based Message Authentication Code\|HMAC]]-[[SHA-256]]) | Shared secret           | Anyone with the shared secret | Anyone with the same shared secret                              |
| ==Digital signature==           |                                     `RS256`, `PS256`, `ES256`, `EdDSA` | Private/public key pair | Only holder of private key    | Anyone with trusted public key (e.g. [[JSON Web Key Set\|JWK]]) |
Q: Seems to me like you'd want to use the latter for a multi-service architecture where you have services authenticating incoming JWTs using a JWKS? So that you don't have to share the symmetric key around.
A: Yes. This is why asymmetric JWT signing is usually the better architectural default when an authentication service issues tokens that many resource services need to verify. If you have a monolithic architecture, then an HMAC solution might be sufficient, but if you have a microservices architecture it's likely better to go the asymmetric route.


Algo lookup:

| JWT alg | Full name                                                          | Key type                       | Hash                        | Modern status                                                            |
| ------- | ------------------------------------------------------------------ | ------------------------------ | --------------------------- | ------------------------------------------------------------------------ |
| `RS256` | `RSASSA-PKCS1-v1_5` using SHA-256                                  | [[Rivest-Shamir-Adleman\|RSA]] | [[SHA-256]]                 | Common and widely supported; uses an older RSA padding scheme            |
| `PS256` | `RSASSA-PSS` using SHA-256 and `MGF1` with SHA-256                 | [[Rivest-Shamir-Adleman\|RSA]] | [[SHA-256]]                 | Preferred modern RSA signature scheme                                    |
| `ES256` | ECDSA using the P-256 elliptic curve and SHA-256                   | Elliptic curve                 | [[SHA-256]]                 | Common; smaller signatures than RSA; implementation details are trickier |
| `EdDSA` | Edwards-curve Digital Signature Algorithm, usually Ed25519 in JWTs | Edwards elliptic curve         | Built into the EdDSA design | Modern, simple, fast, and deterministic                                  |





