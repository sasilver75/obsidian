---
aliases:
  - WebSocket
---
A protocol for full-duplex persistent connection between a client (usually a browser) and a server over a single long-lived [[Transport Control Protocol|TCP]] connection.

How it works:
1. Handshake: Client sends an HTTP `Upgrade: websocket` request, and a Server responds with `101 Switching Protocols`. After this, the connection is no longe [[HTTP]]; it's the [[WebSockets|WebSocket]] wire protocol over the same TCP socket.
2. Framed messages: Both sides send binary or text frames. Tiny message overhead (2-14 bytes) vs HTTP's headers-on-every-request.
3. Bidirectional:  Server can push to client whatever (no polling, no long-polling hacks)
4. Stateful: Connection stays open, so the server typically holds per-connection state (user ID, room, subscription).

In Browser:
```javascript
const ws = new
WebSocket('wss://example.com/socket');
ws.onmessage = (e) => console.log(e.data);
ws.send('hello');
```

Use cases:
- Chat/messaging
- Live dashboards, stock tickers, sports scores
	- ((It seems to me that this would be a better case for the unidirectional [[Server-Sent Event]] (SSE)))
- Collaborative editing (Figma, Google Docs)
- Multiplayer games
- Real-time notifications

Tradeoffs:
- ✅: Low-latency, low-overhead, true bidirectional, widely supported in browsers.
- ❌: Stateful: Doesn't fit [[Serverless]] models well. 
	- Requires [[Sticky Session]]s behind [[Load Balancing|Load Balancer]]s.
	- Scaling means coordinate connections accross servers.
- ❌: No built-in reconnect, auth refresh, or message ordering guarantees; you have to build this.
- ❌: Proxies/firewalls sometimes interfere; long-running connections cost more than short HTTP requests on some platforms.
# Alternatives
- [[Server-Sent Event]] (SSE): Server -> client only, simpler, runs over plain HTTP. Good when you don't need client -> server streaming.
- [[Long Polling]]: Fallback for environments that block WebSockets.
- [[WebTransport]]: A newer bidirectional protocol built on [[HTTP 3]] + [[QUIC]], supporting unreliable datagrams (good for games). Not yet universal.
- Higher-level libs: Socket.IO (adds reconnect, rooms, fallbacks, etc.)


# Why is it better to use a [[Transport Layer|Layer 4]] Load Balancer for WebSockets?
- WebSockets start as [[HTTP]], but then upgrade into a long-lived bidirectional connection. After the upgrade, the connection is no longer normal request/response HTTP traffic: It becomes a persistent TCP connection carrying WebSocket frames.
- This creates an issue for [[Application Layer|Layer 7]] load balancers:
	- Connections must stay open for long time
	- LB must maintain connection state
	- Request-level routing matters only at connection setup
	- There may be no normal HTTP requests after upgrade
	- Some L7 features like per-request routing, buffering, retries, response integration don't apply well
	- ((Basically, L7 are typically based on routing using HTTP features; WebSockets, other than the setup, aren't HTTP -- they're another application layer protocol))
- An L4 load balancer is a good fit, because WebSockets are ultimately long-lived TCP connections. The L4 LB just keeps the TCP flow mapped to one backend; it doesn't need ot understand every WebSocket message.


_____________


In cases where we have high-frequency updates and bidirectional communications. They're very powerful, but also require a lot of infrastructure; you might want to reach for a [[Polling]] or [[Server-Sent Event|SSE]] solution before reaching for a WS solution, especially if you don't need the bidirectionality (e.g. getting an update on a bid for a browser of a page).

In the cases where you need them, they're very powerful! 

A way for you to exchange binary blobs that come in on the same order... and are in some sense guaranteed to be delivered reliably.

In a system design interview, you typically talk about an API in terms of ==messages== that you're sending.
- Becauese these aren't request and response, you don't typically have the input/output setup... instead, you say:

![[Pasted image 20260605165549.png]]
It doesn't eliminate the possibility that you have a message like "subscription accepted"... but you shouldn't design your API in with the expectation of having a bunch of request/response style communications. If you want that request/response functionality, just use HTTP.


### Challenge!
- Websockets in general involve a lot of ==state== for your application.
- If you've got a WS connection, and that user is going to be around for an hour, I need a way for keeping that server alive for as long as that connection needs to be alive. 
	- This can play havoc if you're likely to have ==failures==, or if you want to do ==deployments of software==.
	- We typically want to minimize statefulness, but websockets are inherently stateful!
- ==The way we handle in an interview session is by having something on the periphery or edge of our design that handle the websockets and exposes methods that internal services can call.==
	- You'll have your users connect via webosckets to that service, and then that service will make requests to your internal services, which will then send messages back.
	- See the HelloInterview realtime updates guide.



_____________

Q: What does it look like in practice to have a long-lived websocket connection to an application server that's behind a load balancer, api gateway, etc?

WebSockets are managed differently from normal HTTP because after the initial request, the [[Transport Control Protocol|TCP]] connection stays open and becomes stateful.

Initial connection: Browser starts with an [[HTTP]] request:
```http
GET /chat HTTP/1.1
Host: api.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: ..
```
If accepted, the server responds: `101 Switching Protocols`

After that, it's no longer normal request/response HTTP; instead, the *same underlying TCP connection* now carries [[WebSockets|WebSocket]] frames!

The flow:
- browser -> [[Transport Control Protocol|TCP]] Handshake
- browser -> [[Transport Layer Security|TLS]] Handshake
- browser-> HTTP upgrade request (above)
- server -> Switching protocols
- Nowwwww WebSocket frames can flow over the same connection.

Okay, so how does this work in the scenario that I asked about?
Let's say we're in a scenario of:
`Browser -> Load Balancer -> App Server Pool`


#### Where termination happens
- There are three layers to think about:
	- TCP Termination: Can happen at the LB, which then opens a separate TCP connection to an app server. Some L4 load balancers mostly forward flows, but operationally there's still connection tracking and backend selection.
	- TLS Termination: Can terminate at:
		- The load balancer, with plain HTTP/WebSocket to the backend
		- The load balancer, then re-encrypted TLS to the backend
		- The app server directly, if the load balancer does TLS passthrough (e.g. L4 load balancer)
	- WebSocket Termination:
		- Happens wherever the WebSocket protocol is actually handled.
			- If the app server reads/writes WebSocket frames, the app terminates WebSocket.
			- If a dedicated realtime gateway handles WebSockets and talks to the app services itself via gRPC, then the gateway terminates WebSocket.

#### Does the App Server hold the socket?
- If WebSocket terminates at the app server: Yes!
- If there is a separate WebSocket gateway, then the app server may hold no socket at all.

A WS connection is pinned at connection establishment.
- The LB chooses an application instance using [[Round Robin]], [[Least Connections]], [[Consistent Hashing]], etc.
- There is no per-message load balancing for a single WebSocket connection. WebSocket messages are not *separate HTTP requests,* they aer frames inside one persistent connection.

Each WS-handling node keeps an in-memory registry, for example:
`user_id -> connection_ids -> socket objects`

So when user `123` connects to app server A, server A records:
`user:123 is connected on server A via socket abc`



The key interview answer is: the WebSocket is a long-lived upgraded HTTP connection. The load balancer chooses one backend when the connection starts, and that backend owns the connection until it closes. To send messages from other servers, production systems use shared routing state plus a broker, or they centralize socket ownership in dedicated realtime gateway nodes.

____________
# How do deployments work when you have stateful connections like WebSockets?

Typically, during deploys, good systems drain connections:
1. Remove instance from load balancer rotation
2. Stop accepting new WS connections
3. Let existing sockets continue for some grace period
4. Optionally send clients a close frame with a reason like "server restarting."
5. Client reconnect and land on healthy new instances
6. Kill the old process after the drain timeout.

Without draining, deploys would cause mass disconnects, which can be acceptable for some apps, but realtime-heavy systems usually manage it carefully.

-------------------------


# Idle Timeouts and Reconnects
- LBs and proxies often close idle connections, For example, if no bytes pass for 60 seconds, the proxy may close the connection, even though the browser and server still think it's alive.
- Some WebSocket systems avoid this by using heartbeats. This heartbeat interval must be shorter than the lowest idle timeout in the path, whether that's from the load balancer, proxy, CDN, NAT, firewall, etc. This proxy just looks like a serving ping and client pong being sent between client and server.


# Reconnects
- Reconnects are normal.
- On reconnect, the client usually:
	1. Opens a new WS connection
	2. Authenticates again using cookie/token/signed URL
	3. Resubscribes to rooms/channels
	4. Sends its last seen message ID or cursor
	5. Sever sends missed messages or tells the client to refetch state

If connection drops while messages are sent, those messages can be lost unless the system hs persistence, sequence numbers, acknowledgements, or replay.



__________

# How does Authentication work for WebSockets?
- Typically happens ==during the WebSocket handshake, which starts as a normal HTTP request!==
```http
GET /ws HTTP/1.1
Host: api.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: ...
Sec-WebSocket-Version: 13
Cookie: session=abc123
```
The server can authenticate that HTTP upgrade request before accepting the WebSocket.
==The server then attaches the identity the live connection, and future messages use that connection identity.==

So after authn, the server has connection state like:
```
socket_id: conn_abc
user_id: user_123
tenant_id: tenant_456
roles: ["member"]
subscriptions: []
authenticated_at: 2026-06-07T...
expires_at: ...
```
So when a message arrives:
```json
{
	"type": "send_message",
	"room_id": "room_789",
	"body": "hello"
}
```
The server does NOT trust the client to say who the user is. It uses the identity already attached to the socket.
- "This Frame came from `conn_abc`. `conn_abc` belongs to `user_123`, there fore this action is being attempted by `user_123`"

##### Common auth methods:

##### 1) `Cookie-Based Auth`:
Very common for ws. If a browser has a valid cookie for `api.example.com`, it automatically includes that cookie in the ws handshake, subject to normal cookie rules:
```
GET /ws HTTP/1.1
Host: api.example.com
Upgrade: websocket
Cookie: session=s%3Aabc123...
```
The server validates the session cookie the same way it would for HTTP:
- Read cookie
- Verify signature, look up session
- Identify user
- Accept or reject upgrade
This is clean for browsers, because the native browser `WebSocket` API ==does not let JS set arbitrary heads like `Authorization`!==

##### 2) `JWT in Query String`
- Because Browser cannot set custom headers using the browser's native WebSocket API, another common pattern is:
```
new WebSocket("wss://api.example.com/ws?token=eyJ...")
```
This handshake becomes:
```
GET /ws?token=eyJ... HTTP/1.1
Host: api.example.com
Upgrade: websocket
```
The server extracts and validates the token before accepting the connection.

This has tradeoffs:
- URLs can appear in logs or debugging tools
- Long-lived JWTs in URLs are risky!
- Prefer short-lived, single-purpose WebSocket Tokens.

A better version is:
  1. Client calls HTTPS endpoint with normal auth.
  2. Server returns short-lived WebSocket ticket.
  3. Client connects to wss://api.example.com/ws?ticket=...
  4. Server validates ticket and upgrades connection.

##### 3) `Signed URL`
- Similar to query-token auth, but the URL itself is signed:
```
wss://api.example.com/ws?user=123&expires=...&signature=...
```
Server checks whether the signature is valid, expiration is in the future, parameters not tampered with.
This is common when another service authorizes access to a realtime stream.


##### 4) `Authoriztion Header`
- This is common for non-browser clients (mobile apps, backend services, CLI clients)
```
GET /ws HTTP/1.1
Host: api.example.com
Upgrade: websocket
Authorization: Bearer eyJ...
```
This is usually unavailable for browser-native WS, unless you're using an environment or wrapper that supports custmo headers outside the standard browser API.
