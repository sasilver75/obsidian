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


