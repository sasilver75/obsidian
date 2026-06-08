---
aliases:
  - Authn
  - Authenticate
---
The process of verifying that someone or something is who they claim to be. 
- Answers: "Who are you?"
- Is different from [[Authorization]], which asks: "What are you allowed to do?"


# Common Approaches
- Server [[Session]] + [[Cookie]] (Session Mechanism)
	- How it works: User logs in, server stores a session record, browser gets a session ID cookie.
	- +: Traditional web apps, Next/Rails/Django/Laravel apps, admin panels
	- -: Easy revocation, simple security model, but requires server-side session storage
- [[JSON Web Token|JWT]] User Token: (Session Mechanism)
	- Server [[Cryptographic Signature|Sign]]s a JSON token containing user claims. Client sends it with requests. A [[JSON Web Key Set|JSON Web Key]] is a key represented as JSON. [[JSON Web Key Set|JWKS]] are a set/list of JWKs, published at a url, and are how a JWT verifier gets the keys needed to check that a JWT's signature is real. Why is there more than one JWK in a JWKs? Because signing keys need to rotate without breaking existing tokens.
	- +: APIs, microservices, short-lived access tokens
	- -: Stateless and portable, so harder to revoke before expiry; easy to misuse
- [[Opaque Token]]: (Session Mechanism)
	- Client receives a random token; server/database looks it up. It might look like `Authorization: Bearer 8f3b1c9a4...`. The API can't learn anything from the token text alone, it has to look it up (e.g. in Redis). Typically the server stores a hash of the opaque bearer token itself. If a user logs out, you delete/revoke the token. If permissions change, the next look sees new permissions. If an account is banned, the token stops working. Very conceptually similar to a session cookie (which itself is basically an opaque token).
	- +: APIs where revocation matters
	- -: Requires lookup, but safer and simpler than JWT in many cases.
- [[OAuth|OAuth 2.0]] (Federation Protocol)
	- Delegated authorization: "Allow this app to access X.". "Login with Google". Note: OAuth alone isn't really "login," you use OIDC for identity.
- [[OpenID Connect]] (OIDC) (Federation Protocol)
	- OAuth 2.0 plus identity layer. Gives you a verified user identity
	- +: "Sign in with Google", enterprise [[Single Sign-On|SSO]], external identity providers
	- -: More moving parts, but standard and widely supported
- [[Security Assertion Markdown Language|SAML]] (Federation Protocol)
	- XML-based enterprise SSO protocol
	- +: Older/large enterprise integrations
	- -: Powerful but verbose and painful compared to OIDC
- API Keys (Machine Auth)
	- Static secret sent with API calls
	- +: Okay to server-to-server access, developer APIs
	- -: Usually identifies an app, rather than an end user; needs rotation and scoping.
- Basic Auth
	- Username/password sent with each request, usually base64 encoded, over [[HTTPS]]. 
	- +: Simple internal tools, legacy systems
	- -: Only acceptable over HTTPs, not ideal for modern user auth. Password is sent repeatedly, browser cache the credentials awkwardly, weak session control, higher blast radius.
- Magic links/One-Time Password (OTP) (Credential Method)
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


# [[JSON Web Token|JWT]] vs [[Session]]
- With a session, the cookie usually contains a random ID:
```
cookie: session_id=abc123
server: looks up abc123 in Database/Redis
```

- With a JWT, the token contains signed user data:
```
Authorization: Bearer eyJ...
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

Q: Why don't we just make our access tokens long-lived?
A: Because if an access token with a short-lived expiry is somehow leaked, the damage window is short! Whereas if a long-lived access token were to leak that would be more serious. We're able to use longer-lived refresh tokens for convenience because we can always revoke them on the server side if needed.


