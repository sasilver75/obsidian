---
aliases:
  - Authn
  - Authenticate
---
The process of verifying that someone or something is who they claim to be. 
- Answers: "Who are you?"
- Is different from [[Authorization]], which asks: "What are you allowed to do?"


# Common Approaches
- Server [[Session]] + [[Opaque Token]] stored in client [[Cookie]]: (Session Mechanism)
	- How it works: User logs in, server stores a session record, browser gets a session ID cookie.
	- Client receives a random token; server/database looks it up. It might look like `Authorization: Bearer 8f3b1c9a4...`. The API can't learn anything from the token text alone, it has to look it up (e.g. in Redis). Typically the server stores a hash of the opaque bearer token itself. If a user logs out, you delete/revoke the token. If permissions change, the next look sees new permissions. If an account is banned, the token stops working. Very conceptually similar to a session cookie (which itself is basically an opaque token).
	- +: Traditional web apps, Next/Rails/Django/Laravel apps, admin panels. APIs where revocation matters
	- -: Requires server storage and lookup, but safer and simpler than JWT in many cases. 
- [[JSON Web Token|JWT]] User Token: (Session Mechanism)
	- Server [[Cryptographic Signature|Sign]]s a JSON token containing user claims. Client sends it with requests. A [[JSON Web Key Set|JSON Web Key]] is a key represented as JSON. [[JSON Web Key Set|JWKS]] are a set/list of JWKs, published at a url, and are how a JWT verifier gets the keys needed to check that a JWT's signature is real. Why is there more than one JWK in a JWKs? Because signing keys need to rotate without breaking existing tokens.
	- Stateless, so you can validate credentials without needing a database (e.g. of user sessions), making it perhaps more easily scalable.
	- +: APIs, microservices, short-lived access tokens
	- -: Stateless and portable, so harder to revoke before expiry; easy to misuse
- [[OAuth|OAuth 2.0]] (Federation Protocol)
	- Delegated *authorization*: "Allow this app to access X.". "Login with Google". Note: OAuth alone isn't really "login," you use OIDC for identity.
	- Answers: "What can we access on behalf of the user?"
		- 
- [[OpenID Connect]] (OIDC) (Federation Protocol)
	- OAuth 2.0 plus identity layer. Gives you a verified user identity
	- +: "Sign in with Google", enterprise [[Single Sign-On|SSO]], external identity providers
	- -: More moving parts, but standard and widely supported
- [[Security Assertion Markdown Language|SAML]] (Federation Protocol)
	- XML-based enterprise SSO protocol
	- +: Older/large enterprise integrations
	- -: Powerful but verbose and painful compared to OIDC
- [[API Key Authentication]] (Machine Auth)
	- Generate a unique key for each client, and they send it with each request to access the resources.
		- Typically in either an `Authorization: ApiKey abc123` or `X-Api-Key: abc123` header
			- Typically stored in service backend with a table with `id, key_hash, user_id, scopes, active_bool`
			- Service does an API key lookup in this table, can verify if valid.
	- Q: How is this different from the Session+OpaqueToken+Cookies approach? It seems that both involve an opaque token being sent by the client, and then the server storing information about what that token means. The header it's sent in may be different, but ....
		- A: You're basically right that the mechanics are very similar in the sense of server-stored state and user-supplied opaque auth tokens. The difference is mostly semantics/lifecycle/security expectations.
			- User session token: Represents a human user logic, crated after interactive authentication (password, [[Single Sign-On|SSO]], [[Multi-Factor Authentication|MFA]], passkey), often short-lived and tied to browser/device metadata, and revoked by logout/password reset/admin action. Usually stored in an`HttpOnly SameSite Secure` cookie, for browsers. Answers "Which user is currently logged in from this client?"
			- API Key: Represents an application, integration, service account, project, developer credential. Usually created manually/programmatically, not through an interactive login per request. Often longer-lived than a user session, and designed for scripts, servers, CLIs, integrations, backend-to-backend calls. 
			- Typically a Session token principal is usually `session_id+user_id`, while an API key principal is often `api_key_id + app_id/org_id/service_account_id`. Sessions are ephemeral login state, while API keys are durable credentials meant to be rotated/scoped/audited and used non-interactively.
	- +: Okay to server-to-server access, developer APIs
	- -: Usually identifies an app, rather than an end user; needs rotation and scoping. If key leaks, people can replay it (similar to JWT). 
- [[Basic Authentication]]
	- Username/password sent with each request, usually [[Base64]] encoded, over [[HTTPS]]. 
	- +: Simple internal tools, legacy systems
	- -: Only acceptable over HTTPs, not ideal for modern user auth. Password is sent repeatedly, browser cache the credentials awkwardly, weak session control, higher blast radius.
	- Outdated and rarely used today
- [[Digest Authentication]]
	- Basically just Basic Auth but you [[MD5]] hash your password in the payload instead of just Base64 encoding it.
	- Outdated and rarely used today
- Magic links/[[One-Time Password]] (OTP) (Credential Method)
	- User gets a one-time link/code by email/SMS
	- +: Passwordless login!
	- -: Email/SMS becomes the security boundary; SMS is weaker.
- Passkeys/[[Web Authentication API|WebAuthn]]
	- Cryptographic login using device/private key
	- +: Modern passwordless user authentication
	- -: Excellent security, but UX and compatibility need care.
- [[Mutual TLS|mTLS]] (Machine Auth)
	- Both server *and client* prove identity with a [[Certificate]].
	- +: High-trust service-to-service systems, often combined with [[Service Token]]s for [[Authorization|Authz]]
	- -: Strong, but operationally heavier.


# [[Access Token]] vs [[Refresh Token]]
- Access Tokens are short-lived, and used for API calls to the server.
- Refresh Tokens are long-lived, and are used to get new Access Tokens (to "renew" the Access Token).
- The user gets both of these tokens on signin
	- e.g. Access token valid for 15min-1hour, and a Refresh token accessible for 7h-30d
- Client uses the access token to access the API and make requests, and stores the Refresh token, commonly in [[Cookie]]s (`HttpOnly`, `SameSite`, `Secure`). The guards it against [[Cross-Site Scripting|XSS]] attacks.
- Via sending along the Access token with request stays logged in without re-entering credentials...
- Eventually, they will get an Authorized response when their access token expires; at that point, the client makes a request to the auth server with their Refresh Token, which will return a new JWT Acces Token.



# [[JSON Web Token|JWT]] vs [[Session]]
- With a session, the cookie usually contains a random Session ID:
```
cookie: session_id=f4o1ifn4molkf1...
server: looks up f4o1ifn4molkf1... in Database/Redis, and gets something back like:

{
	"user_id": "user_42",
	"expires_at": "...",
	"roles": ["admin"]
}
```

- With a JWT, the token contains signed user data:
```
Authorization: Bearer eyJ...

This eyJ... actually looks something like eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWUsImlhdCI6MTUxNjIzOTAyMn0.KMUFsIDTnFmyG3nMiGM6H9FNFUROf3wh7SmqJp-QV30

It has three parts: header, payload, and signature, which are period-delimited. 
All are Base-64 encoded

server: verifies signature and reads claims
```


The key differences:
- Sessions are stateful: The server stores session data
- JWTs are stateless: server can validate the token without a database lookup

This sounds like JWTs are automatically better, but they aren't! JWTs are harder to revoke, often leak too much data, and can create stale permission problems. ==For many normal web apps, secure cookie sessions are the better default==.

For a normal web application:
- Use server-side sessions
- Store the session ID in a `Secure`, `HTTPOnly`, `SameSite` cookie
- Keep session data on the server
- Add [[Cross-Site Request Forgery|CSRF]] protection when needed
- Use [[OpenID Connect|OIDC]] if you want "Sign in with Google/Github/Okta/etc."

For an API:
- Use short-lived access tokens
- Prefer opaque tokens if you need easy revocation
- Use [[JSON Web Token|JWT]]s when services need to validate tokens without a central lookup.
- Use [[Refresh Token]]s carefully, usually stored in secure cookies or protected storage.

For service-to-service auth:
- Use API keys, OAuth client credentials, [[Mutual TLS|mTLS]], or signed requests 

==Short takeaway==:
- Sessions for web applications
- OIDC for third-party login/SSO
- Opaque tokens for revocable APIs
- JWTs for short-lived distributed API authentication

Q: We've made the claim that form any normal web apps, secure cookie sessions are the better default. Why is this, exactly?
A: Basically because the server keeps the real auth state. You can revoke/logout a user, do role changes, handle suspicious sessions, etc immediately by deleting/updating server-side state. With a self-contained JWT, the if the signature is valid and the token is not expired, it's often accepted.


____________________

# [[Refresh Token]]s
- A Refresh Token is a *long-lived credential* that is used to get new short-lived access tokens!
- It's not usually sent to your API on every request, it's sent only tot he auth server/token endpoint.

Flow:
1. User logs in
2. Auth server returns:
	1. Access token (expires in ~5-60 minutes)
	2. Refresh Token (expires later or after inactivity)
3. Client calls API with access token
4. Access token expires
5. Client sends refresh token to auth server
6. Auth server returns a new access token

Access tokens are often [[JSON Web Token|JWT]]s or similar: `Authorization: Bearer short_lived_jwt`

Refresh tokens are often [[Opaque Token]]s, even in systems where the access token is a JWT: `refresh_token=random_secret_string`.

This is useful because refresh tokens need revocation, rotation, reuse detection, and server-side control!

So a very common combination is:
- JWT access token gives fast API validation (because servers can validate without network requests, using an appropriate JWK)
- Opaque refresh token gives the auth server control, allowing it to revoke it if needed.

Q: Why don't we just make our *access* tokens long-lived?
A: Because if an access token with a short-lived expiry is somehow leaked, the damage window is short! Whereas if a long-lived access token were to leak that would be more serious. We're able to use longer-lived refresh tokens for convenience because we can always revoke them on the server side if needed.


_________




