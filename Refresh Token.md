
A longer-lived credential used to get a new short-lived (e.g.) [[Access Token]] without making the user log in again. It is used only to get a new short-lived access token when the old access token expires or is about to expire.
- Note: Typically used for token0based systems (JWTs, OAuth 2.0, OIDC), while traditional opaque-token cookie-based server sessions usually do not need a separate refresh token.

We need refresh tokens so that access tokens can stay short-lived, so that if an access token leaks, it stops working soon, while the refresh token can be protected more tightly, revoked, rotated, and used only to issue new access tokens.

Refresh tokens are usually [[Opaque Token]]s (a long random string also stored as a hashed copy server-side). When the client sends the refresh token, the server looks it up and decides whether to issue a new access token. Opaque tokens make it:
- Easier to revoke
- Easier to rotate
- Easier to detect reuse
- No sensitive claims exposed to the client
- No need for the token to be self-verifiable by APIs

A JWT refresh token is possible, but is less common for normal apps, because refresh tokens are usually only sent to the auth server anyways. Since the auth server can look them up, stateless verification is less valuable.

### 1) User logs in

User sends login request
```http
POST /auth/login
Content-Type: application/json

{
	"email": "sam@example.com",
	"password": "correct-horse"
}
```

Server responds
```http
HTTP/1.1 200 OK
Set-Cookie: refresh_token=r_abc123; HttpOnly; Secure; SameSite=Lax; Path=/auth/refresh; Max-Age=2592000
Content-Type: application/json

{
	"access_token": "eyJhbGciOi...",
	"token_type": "Bearer",
	"expires_in": 900
}
```
Above: 
- The Server can commonly return either (a JWT access token or an Opaque access token) in either (the response body, or a cookie). Server setting multiple cookies on client just looks like the same `Set-Cookie` header multiple times. Here, we chose to put it in the response. 
	- Typically, if the server returns the access token in the response body, that usually means that the client app itself is expected to hold it and send it in an `Authorization` header. In a browser [[Single Page Application|SPA]], the least-bad place is usually memory-only. JavaScript-created cookies cannot be `HttpOnly`, which means [[Cross-Site Scripting|XSS]] can read the cookie.
	- Rough rule:
		- Access token returned in body? Store in memory, use Authorization header
		- Access token should be in cookie? Server sets `HttpOnly Secure SameSite` cookie.
		- AVOID storing access tokens in [[Local Storage]] unless you have a specific reason and understand the XSS tradeoff.

### 2) Client uses Access Token for API Requests
Example: Fetch invoices
```http
GET /api/invoices
Authorization: Bearer eyJhbGciOi...
```
Example: Create a project
```http
POST /api/projects
Authorization: Bearer eyJhbGciOi...
Content-Type: application/json

{
"name": "New Project"
}
```

Along the way, the access token is what proves that the request is authenticated; The server might verify the signature on the (commonly, if sent by `Authorization` header) JWT, check it's not expired, look at the claims, and decide to authorize the request.

### 3) Client knows when the Access Token is expiring

Two common strategies:
- Proactive Refresh (possible with either JWT or Opaque Session Token cookie)
	- The Access token says that it expires at 12:15:00
	- Client refreshes around 12:14:00
- Reactive Refresh
	- Client makes an API request
	- Server says 401 Unauthorized, since the token/session has expired
	- Client request a new access token
	- Client retries original request

In the reactive case, client might receive this response:
```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer error="invalid_token", error_description="access token expired"
```

### 4) Client requests a new Access Token
- Because the refresh token is in an`HttpOnly` cookie, JS does not read it; the browser sends it automatically, only to the matching path.

```http
POST /auth/refresh
Cookie: refresh_token=r_abc123
```

The server validates the refresh token; if valid, it returns a new access token:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
	"access_token": "eyJhbGciOi_new...",
	"token_type": "Bearer",
	"expires_in": 900
}
```

If using refresh [[Key Rotation|Token Rotation]], the server might also issue a new refresh token in the response:
```
Set-Cookie: refresh_token=r_def456; HttpOnly; Secure; SameSite=Lax; Path=/auth/refresh; Max-Age=...
```

### 5) Client Retries the original request

```
GET /api/invoices
Authorization: Bearer eyJhbGciOi_new...
```


So the access token does the normal work, and the refresh token is mainly for renewing access.

Because refresh tokens are powerful, they should be protected carefully! Commonly, they're stored in `HttpOnly Secure Samesite`.


_____________

Q: Wait, cookies are automatically sent, no? Wouldn't that mean we're always  sending our refresh token?
A: Sort of. Cookies are automatically sent with ==matching requests,== but "matching" is load-bearing:
- Domain: Is this request going to the cookie's domain?
- Path: Does the request path match?
- Secure: Is the request HTTPS?
- SameSite: Is this same-site or cross-site?
- Expiration: Is the cookie still valid?

So if you set:
`Set-Cookie: refresh_token=abc123; HttpOnly; Secure; SameSite=Lax; Path=/auth/refresh`
- Then the browser only sends it to `https://example.com/auth/refresh`, but not to `https://example.com/api/orders`, because the path does not match.


Aside: `SameSite=Lax` is a common default, because it gives you decent [[Cross-Site Request Forgery|CSRF]] protection while still allowing normal navigation into your site. With this setting, the browser sends the cookie on:
- Same-site requests, like your app calling your API on the same site
- Top-level navigation from another site, like clicking a link to https://example.com
But it doesn't send the cookie on cross-site background/subresource requests, like:
- A hidden form `POST` from another site
- An image tag loading your URL
- A cross-site `fetch()`
- An `iframe` request in many cases
So an attacker's site has a harder time making the victim's browser send your auth cookie to perform actions.
In contrast, `SameSite=Strict` give stronger CSRF protection, but can break normal flows (e.g. clicking a link from email into your app, they cookie won't be sent on first navigation, which can look logged out until they reload or navigate).





