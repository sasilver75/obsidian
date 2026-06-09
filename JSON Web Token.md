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


# How do we issue JWTs?
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
- SPAs with careful token handling
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















