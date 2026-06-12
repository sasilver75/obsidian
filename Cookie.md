A type of [[Browser Storage]].

Cookies are small pieces of string data that browsers can automatically attach to matching [[HTTP]] requests.

- Unlike [[Local Storage]] or [[Session Storage]], cookies can be sent to the server automatically.
- Usually small, commonly around 4KB each.
- Commonly used for login [[Session]]s, server-side [[Authentication]], personalization, tracking, and analytics.
- Tracking and analytics cookies are heavily regulated and restricted by modern browsers.

Important cookie attributes:
- `Secure`: only sent over [[HTTPS]].
- `SameSite`: controls cross-site sending behavior.
- `HttpOnly`: prevents JavaScript access.
- `Expires` / `Max-Age`: controls lifetime.

For [[Authentication]], a secure `HttpOnly` cookie is usually safer than putting a token in [[Local Storage]], because [[Cross-Site Scripting|XSS]] cannot directly read it.


# Use with [[Authentication Tag]]s
- For an opaque server-side session token used for [[Authentication]], there's no need to use an authentication tag.
```
__Host-session=6Lq9...random...T2w
```
- In this case, the cookie has no meaning by itself; the server uses (a hash of) it as a database/cache key. 
- The security come from the identifier being unpredictable, unique, and validated against server-side session state.

- In the case where you're actually storing data in a cookie, for example:
```
user_id=123&role=admin&expires=...
```
- If the server accepts this kind of cookie, the server must authenticate it, usually with an [[Hash-based Message Authentication Code|HMAC]]-based [[Authentication Tag]]... otherwise the user can edit `role=user` into `role=admin`!
In this case, we would do:
```
cookie_value = base64url(payload) + "." + base64url(hmac(secret_key, payload))
```
concretely, maybe:
```
Set-Cookie: app_session=eyJ2IjoxLCJ1c2VyX2lkIjoidXNlcl8xMjMiLCJpc3N1ZWRfYXQiOjE3ODEyOTAwMDAsImV4cGlyZXNfYXQiOjE3ODEyOTM2MDAsInNlc3Npb25fdmVyc2lvbiI6N30.2uI7z7nYVjXGd9n6Z3pF9cM7qN7Kf9wUeK2cYjTqM_g; Path=/; Secure; HttpOnly; SameSite=Lax
```
Conceptually, the first part is readable JSON after [[Base64]] URL decoding:
```json
{
  "v": 1,
  "purpose": "cart",
  "items": [
    { "product_id": "sku_123", "quantity": 2 },
    { "product_id": "sku_987", "quantity": 1 }
  ],
  "coupon_code": "SUMMER10",
  "expires_at": 1781293600
}
```
Cookie format is then:
```
v1.base64url(payload).base64url(HMAC-SHA-256(secret_key, "v1." + base64url(payload)))
```
The second part is the [[Message Authentication Code|MAC]], computed commonly with something like [[Hash-based Message Authentication Code|HMAC]]-[[SHA-256]].

On receipt, the server:
1. Split cookie into payload and received HMAC.
2. Recompute expected HMAC over the exact payload bytes.
3. Compare received HMAC and expected HMAC using constant-time comparison.
4. If valid, decode payload.
5. Check expires_at, issuer/version, and any revocation/session-version rules.

So if the user maliciously tries to change the cart contents from `"quantity": 2` to `"quantity": 200`, the HMAC verification fails unless the attacker knows the server's HMAC secret key (which in this case should never leave the cookie-authenticating server).

This example is somewhat JWT-like in the broad sense that it's a signed, Base64URL-encoded JSON. Difference is that a JWT is a standardized token format with a header, registered claim names, algorithms, and ecosystem conventions. A signed application cookie is just your own serialized application state plus an Authentication Tag.