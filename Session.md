---
aliases:
  - User Session
---
A server-side or client-visible continuity of a user's authenticated interaction across multiple (typically stateless [[HTTP]]) requests. 
- By default, the server doesn't know if two successive requests came from the same logged-in user. A session mechanism adds continuity!


# 1) Server-Side Sessions
Common Model:
- User Logs In
	- Server verifies credentials
	- Server creates sessionID, stores a hash of it:Session, and tells the client to store the raw SessionID via a `Set-Cookie: session_id=sess_7uW9Xq2mK...; HttpOnly; Secure; SameSite=Lax; Max-Age=28800` header in the response
	- Client stores session identifier, usually in a [[Cookie]]
		- e.g. `session_id=abc123` ; this "abc123" is typically a long random value
		- This cookie typically has a limited lifetime and is revocable.
	- Browser sends cookie on future requests (It comes in the Cookie header: `Cookie: session_id=sess_abc123`. If there are multiple cookies, they just come as `Cookie: session_id=sess_abc123; theme=dark; csrf_token=xyz`)
		- Important flags to set on the server when setting :
			- `HttpOnly`
			- `Secure`
			- `SameSite`: Controls cross-site sending, reducing [[Cross-Site Request Forgery|CSRF]] risk
			- `Expires/Max-Age`: Controls lifetime
			- `Domain/Path`: Controls where cookie sent
	- Server uses session to identify user/request context.
		- Server looks up: `session_id abc123 -> user_id 42, roles, expiration, metadata`

+: Easy to revoke, small cookie, sensitive data stays server-side
-: Requires shared session storage, adds a lookup on each request

The server-side session record that uses the client-side cookie (opaque random id) to look up in (e.g.) Redis might look something like:
```json
{
    "session_id_hash": "sha256:...",
    "user_id": "user_42",
    "created_at": "2026-06-08T10:00:00Z",
    "last_seen_at": "2026-06-08T10:42:00Z",
    "expires_at": "2026-06-08T18:00:00Z",
    "auth_level": "password+mfa",
    "roles": ["admin"],
    "csrf_token_hash": "sha256:...",
    "ip_prefix": "203.0.113.0/24",
    "user_agent_hash": "sha256:..."
  }
```
In practice, on the server-side session store, you often store a [[Hash]] of the session ID (the abc123 opaque token), rather than the raw value itself, so that a database leak doesn't immediately become a list of usable sessions. When a client sends their session ID, the server hashes it and looks up the session in the database.


# 2) Stateless Token Sessions
- Browser stores signed token, often a [[JSON Web Token|JWT]]
- Server validates the token without a DB lookup, using [[JSON Web Key Set]] (JWKS)

+: No central session lookup required, works well in distributed systems
-: Harder to revoke before expiration, token can get large, requires careful expiration and rotation

Note: [[User Access Token]] vs [[Refresh Token]]
- Modern apps often use:
	- Access Token: Short-lived credential for requests
	- Refresh Token: Longer-lived credential used to get new access tokens
- For browser apps, many teams still prefer refresh tokens in secure HttpOnly cookies, rather than localStorage.









