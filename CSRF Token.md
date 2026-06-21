See [[Cross-Site Request Forgery]]

A CSRF token is a server-recognized proof that the request came from code that could read something from the real site, not merely from some other site that could cause the browser to send cookies and cause a [[Cross-Site Request Forgery|CSRF]] attack.

The client usually does not generates the CSRF token. The server generates it, gives it to legitimate client code, and later checks that the client sent it back.

With the introduction of a CSRF token, instead of seeing only:
```http
POST /transfer
Cookie: session_id=abc123
```
The server instead sees:
```http
POST /transfer
Cookie: session_id=abc123
X-CSRF-Token: random-value-known-to-the-bank
```
or, for a more traditional form:
```html
<input type="hidden" name="csrf_token" value="random-value-known-to-the-bank">
```

The malicious site can often cause the browser to send the cookie, but the malicious site usually cannot learn the CSRF token because browser same-origin protections ([[Cross-Origin Resource Sharing|CORS]]) prevent `evil.example` from reading pages or API responses from `bank.example`.

A signed auth token is like a signed badge:
> “This request is from Alice.”

The additional inclusion of a signed *CSRF token* is like a signed form slip:
> “This request includes a form slip I issued to the current logged-in browser session.”

## How does the server Generate and Validate one?
There are a few common designs:
1. ==Synchronizer token==: ==The classic design==. When the server creates a user [[Session]], it also creates a high-entropy random CSRF "token":
	- ```
		session_id = abc123
		csrf_token = 7d94f4...long random value...
	  ```
	- The server stores both of these in server-side session storage, and embeds the CSRF token in pages or responses that legitimate site code can read. For a form UI, this might look like:
	- ```html
		<form method="POST" action="/settings/email">
		  <input type="hidden" name="csrf_token" value="7d94f4...">
		  <input name="email">
		</form>
	  ```
	- When the form is submitted, the server checks:
		- That the automatically attached `session_id` cookie identifies a valid session
		- That the submitted `csrf_token` exists, and matches the token stored for that session
		- That the requested action is allowed by your authorization rules
2. ==Signed token==: In this stateless design, the server might not want to store a CSRF token for every session. Instead, the server can create a token that contains or is bound to session-specific data and is protected by a server secret.
	- Example: `csrf_token = HMAC(server_secret, session_id || nonce, expiration time)`
	- Client receives the CSRF token. Later, when the client submits the CSRF token, the server recomputes or verifies the HMAC.
	- The most important property is that the attacker cannot forge a valid token without the server secret.
	- When received, server validates that the token signature is valid, the token is not expired, etc.
	- ==Note==: For a server-rendered form, this token is usually stored in the page itself:
	- ```html
		<form method="POST" action="/settings/email">
		  <input type="hidden" name="csrf_token" value="signed-token-here">
		</form>
	  ```
	  - For a JS application, it might be stored in a `<meta>` tag, or returned from a same-origin endpoint like `GET /csrf-token` then held in JavaScript memory.
3. ==Double-submit cookie==: The server sends a CSRF token in a cookie and expects the client to send the same value somewhere else, usually a custom request header.
	- Server sends:
		- ```
			Set-Cookie: session_id=abc123; HttpOnly; Secure; SameSite=Lax
			Set-Cookie: csrf_token=random123; Secure; SameSite=Lax
		  ```
	- JS reads the CSRF cookie and sends:
	- ```
	  POST /api/settings/email
	  Cookie: session_id=abc123; csrf_token=random123
	  X-CSRF-Token: random123
	  ```





