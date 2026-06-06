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
- If you've got a WS connection, and that user is going to be around for an hour, I need a way for keeping that server alive for as long as tht connection needs to be alive. 
	- This can play havoc if you're likely to have ==failures==, or if you want to do ==deployments of software==.
	- We typically want to minimize statefulness, but websockets are inherently stateful!
- ==The way we handle in an interview session is by having something on the periphery or edge of our design that handle the websockets and exposes methods that internal services can call.==
	- You'll have your users connect via websocketes to that service, and then that service will make requests to your internal services, which will then send messages back.
	- See the HelloInterview realtime updates guide.








