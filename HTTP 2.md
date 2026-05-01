2015

Originated from Google's experimental SPDY protocol (~2009-2016), but HTTP/2 is the standardized version.

Semantics are unchanged from [[HTTP 1.1]]: Same methods, same status codes, same headers, same caching model.
- What changed is the ==wire format== and ==how connections worked==
- HTTP 2 is great when you have many small assets on one origin (typical web pages, CSS ,JS, fonts, images), gRPC and others multiplexed RPC workloads.
- HTTP 2's losses don't matter when you have one big download (single stream, no multiplexing benefit), etc.

[[gRPC]] runs over HTTP/2 specifically, because it needs:
- Bidirectional streaming
- Trailers (headers sent *after*) body, used for status code after a streaming response
- [[Multiplex|Multiplexing]] (many concurrent RPCs on one connections)


### Core Problem HTTP 2 set out to solve
- [[HTTP 1.1]]'s killer flaw was [[Head-of-Line Blocking]] and no [[Multiplex|Multiplexing]].
	- On a single [[Transport Control Protocol|TCP]] connection, a request #2 had to wait for request #1's response to finish.
		- Browsers would work around this by opening (e.g.) 6 parallel connections per host, and developers worked around this with domain sharding, image spiriting, JS/CSS concatenation, and inlining; all hacks for the same issue.


[[HTTP 2]]'s pitch: 
> *"One connection, many concurrent streams."*

## What Changed
- ==Binary framing==
	- Instead of a text-based, line-oriented framing, used a binary framing layer. Each message is split into typed frames, each with a small header any payload. Frame types include:
		- `HEADERS`: Header block
		- `DATA`: Body bytes
		- `SETTINGS`: Connection-level config
		- `WINDOW_UPDATE`: Flow control
		- `REST_STREAM`: Cancel a stream
		- `PING`, `GOAWAY`, etc
- ==Streams and multiplexing==
	- A stream is a logical bidirectional sequence of frames sharing a stream ID. One TCP connection carries many concurrent streams, interleaved frame-by-frame.
	- ```
	Connection (1 TCP socket)
	  ├── Stream 1: GET /index.html       [HEADERS][DATA][DATA]
	  ├── Stream 3: GET /style.css        [HEADERS][DATA]
	  ├── Stream 5: GET /app.js           [HEADERS][DATA][DATA][DATA]
	  ├── Stream 7: GET /logo.png         [HEADERS][DATA]
	  └── Stream 9: GET /api/me           [HEADERS][DATA]
	```
	- Client-initiated streams use odd IDs, server-initiated ones use even.
	- This eliminates HTTP-level [[Head-of-Line Blocking]] and removes the need for (e.g.) 6 connections per host or domain sharding hacks that you saw with [[HTTP 1.1]]. 
- ==Header compression==: [[HPACK]]
	- [[HTTP 1.1]] used to resend every header as plaintext on ever request. On a cookie-heavy site, a 100-byte request body might carry 2KB of repeated heads.
	- [[HPACK]] is a compression scheme designed for HTTP headers!
		- Static table of common headers (`:method GET`, `:status 200`, `accept-encoding gzip`, etc.) referenced by index
		- Dynamic table built up over the connection: repeated headers get encoded as a single byte.
		- [[Huffman Coding]] for the rest.
	- As a result, header overhead drops dramatically across many requests! This also results in some security caveats (see CRIME-style attacks)
- ==Pseudo-headers==
	- The request line and status line from 1.1 become pseudo-headers prefixed with `::`
	- Header names are required to be lowercase; no more `Context-Type` vs `content-type` ambiguity
```
:method: GET
:scheme: https
:authority: example.com
:path: /api/users/42
accept: application/json

:status: 200
content-type: application/json
```
- ==Flow Control==
	- Per-stream and per-connection flow control via `WINDOW_UPDATE` frames.
	- A slow consumer can stop one stream from drowning the connection without affecting others.
	- This is necessary because ==multiplexing many streams over one TCP socket means a single greedy stream could starve others==.
- ==Stream priorities==
	- Clients could declare dependencies and weights so that the server knew "send the CSS before the hero image."
	- In practice, the priority scheme was complex and inconsistently implemented. 


