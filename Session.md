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
	- Server creates session
	- Client stores session identifier, usually in a [[Cookie]]
		- e.g. `session_id=abc123`
	- Browser sends cookie on future requests
		- Important flags:
			- `HttpOnly`
			- `Secure`
			- `SameSite`: Controls cross-site sending, reducing [[Cross-Site Request Forgery|CSRF]] risk
			- `Expires/Mag-Age`: Controls lifetime
			- `Domain/Path`: Controls where cookie sent
	- Server uses session to identify user/request context.
		- Server looks up: `session_id abc123 -> user_id 42, roles, expiration, metadata`

+: Easy to revoke, small cookie, sensitive data stays server-side
-: Requires shared session storage, adds a lookup on each request


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









