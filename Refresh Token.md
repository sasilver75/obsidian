
A longer-lived credential used to get a new short-lived (e.g.) [[User Access Token]] without making the user log in again.



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
```
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
- See that the Server tells the 


### 2) Client uses Access Token for API Requests


### 3) Client knows when the Access Token is expiring


### 4) Client requests a new Access Token


### 5) Client Retries the original request





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





