---
aliases:
  - HyperText Transfer Protocol
---
HTTP (HyperText Transfer Protocol) is the [[Application Layer]] protocol that the web is built on.
It defines how a client (e.g. browser) and a server exchange messages.

> *"The lingua franca for "ask a server for a thing, or tell it do do a thing. Methods say what you want, URLs say what you're talking about, headers carry metadata, status codes report the outcome, and the body carries the payload. Many other things (REST, GraphQL, webhooks, gRPC, OAuth) are conventions layered on top of these primitives."*

See also:
- [[HTTPS]]
- [[HTTP Range Request]]
- [[HTTP 1.1]]
- [[HTTP 2]]
- [[HTTP 3]]

The core model is ==request/response==.
- A client sends a ==request==, and a server sends back a ==response.== That's it. ==Stateless== by default: the server doesn't inherently remember anything between requests ([[Cookie]]s, [[Session]]s, and [[Token]]s are layered on top, as a way of "faking" state).

### Key Properties
- Stateless: Each request stands alone; [[Session]]s are simulated via [[Cookie]]s/[[Token]]s
- Text-based, originally (in [[HTTP 1.1]]), but now binary ([[HTTP 2]], [[HTTP 3]]), but the *semantics* (methods, status codes, headers) are unchanged across versions.
- Layered: Runs on top of [[Transport Control Protocol|TCP]] ([[HTTP 1.1]] and [[HTTP 2]]) or [[QUIC]] ([[HTTP 3]]).
	- Almost always wrapped in [[Transport Layer Security|TLS]], giving [[HTTPS]], which adds encryption + server authentication via certificates. Essentially all production traffic is HTTPS.
- Cacheable: GET responses can be cached at multiple layers (browser, CDN, reverse proxy). Caching headers can help govern this.
- Extensible: New headers and methods added without breaking existing clients.
- What HTTP is NOT:
	- NOT a Transport, it just sits on top of [[Transport Control Protocol|TCP]]/[[QUIC]]
	- NOT just for browsers; REST APIs, gRPC, webhooks, GraphQL, OAuth, and most of cloud infra speaks HTTP


Request
```
GET /index.html HTTP/1.1
Host: example.com
Accept: text/html
User-Agent: Mozilla/5.0
```

Response
```
Response:
HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 1234

<!DOCTYPE html>...
```


==Methods==:
- GET — fetch a resource (safe, idempotent, cacheable)
- POST — create or submit data (not idempotent)
- PUT — replace a resource (idempotent)
- PATCH — partial update
- DELETE — remove a resource
- HEAD, OPTIONS — metadata / CORS preflight
- Less common: CONNECT (tunneling, used by proxies for HTTPS), TRACE

==Status Codes==:
- 1xx — informational (e.g., 103 Early Hints)
- 2xx — success (200 OK, 201 Created, 204 No Content)
- 3xx — redirect (301 permanent, 302/307 temporary, 304 Not Modified)
- 4xx — client error (400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Too Many
Requests)
- 5xx — server error (500, 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout)

==Headers==: Key-value metadata on requests/responses
- Routing: Host, Referer
- Content negotiation: Accept, Content-Type, Accept-Language
- Caching: Cache-Control, ETag, If-None-Match, Last-Modified
- Auth: Authorization, Cookie, Set-Cookie
- Security: Strict-Transport-Security, Content-Security-Policy, CORS (Access-Control-*)
- Compression: Content-Encoding: gzip|br|zstd

URLs:
```
  https://example.com:443/path/to/resource?query=value#fragment
  └─┬─┘   └────┬────┘ └┬┘└──────┬──────┘ └─────┬─────┘ └───┬──┘
  scheme    host     port     path           query     fragment
```

Body: The Payload
- JSON, HTML, and image, a form upload, a stream, etc.
- Determined by the `Content-Type` header.






















