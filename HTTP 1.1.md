1997

This was the dominant [[HTTP]] protocol for ~20 years and is still everywhere as a fallback.
- If you've ever read raw HTTP request, you've read 1.1
- [[WebSockets|WebSocket]]s is built on it (WebSockets-over-HTTP/2 exists, but is rarely deployed; [[WebTransport]] uses [[HTTP 3]])

> *"HTTP/1.1 is the plain-text, one-request-at-a-time, keep-the-connection-open version. Its limitations (head of line blocking, header repetition, no multiplexing) are what HTTP/2 sets out to fix, and the fact that HTTP/2 STILL couldn't fix TCP-level head-of-line blocking is what produced HTTP/3."*
> *"If you understand 1.1 deeply, you understand HTTP. The newer versions are mostly more efficiently encodings of the same semantics: same methods, same status codes, same headers, same caching model. The wire format and transport layer changed, but the meaning didn't."*

### What 1.1 added over 1.0
- HTTP 1.0 (1996) opened a fresh [[Transport Control Protocol|TCP]] connection for every single request.
	- With pages loading dozens of scripts, images, stylesheets, that meant dozens of TPC handshakes, which sucked for latency.
- 1.1's goal was to make one connection do more work.

### Key Additions
- ==Persistent TCP connections by default== (`Connection: keep-alive`), the connection stays open across requests, and you only pay the [[Transport Control Protocol|TCP]] + [[Transport Layer Security|TLS]] handshake cost once.
	- Still, ==one request must still finish before the next is sent on the same connection==. To get parallelism, browsers open (e.g.) 6 connections per host.
- Better caching via `Cache-Control` header
- Content negotiation via `Accept`, `Accept-Language`, `Accept-Encoding` headers so that the same URL can serve HTLM or JSON, English or Japanese, gzipped or plain.
- [[HTTP Range Request]]s: `Range: bytes=0-1023` for resumable downloads and video seeking. `206 Partial Content`
- More methods: `OPTIONS, PUT, DELETE, TRACE, CONNECT` joined the existing `GET/POST/HEAD`
- More status codes: `100 Continue, 301/303/307, 409 Conflict, 410 Gone, 417 Expectation Failed`, etc.

## Pain Points (why [[HTTP 2]] and [[HTTP 3]] exist)
1. [[Head-of-Line Blocking]] at the HTTP layer: Request 2 sits idle while request 1 finishes, even with a kept-alive TCP connection.
2. No real parallelism without multiple connections (and each new connection means another TCP+TLS handshake)
3. Header bloat: Every request sends the full `Cookie` `User-Agent`, `Accept`, etc., uncompressed. On cookie-heavy sites, the headers can dwarf the body.
4. No prioritization: The server can't be told: "this CSS is critical, send it before that message"

# What 1.1 still does well (or uniquely)
- Trivially debuggable
- Universal
- WebSockets is built on it
- Long-running streams
- ==Most internal service-to-service traffic still speaks 1.1 unless someone deliberately turned on 2 or 3.==


1.1 Example
```
GET /api/users/42 HTTP/1.1
Host: api.example.com
User-Agent: curl/8.0
Accept: application/json
Accept-Encoding: gzip
Connection: keep-aliv
```

```
HTTP/1.1 200 OK
Date: Fri, 01 May 2026 14:23:00 GMT
Content-Type: application/json
Content-Length: 87
Cache-Control: max-age=60
ETag: "a1b2c3"
Connection: keep-alive

{"id":42,"name":"Ada Lovelace","email":"ada@example.com"}
```







