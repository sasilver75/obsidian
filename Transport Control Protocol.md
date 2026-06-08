---
aliases:
  - TCP
  - TCP Handshake
---


Head-of-Line Blocking
TCP Hanshake
TCP Slow Start
TCP Keepalive



![[Pasted image 20260605132139.png]]
- There's a lot of back-and forth
- There's a stateful connection



____________

Q: For a typical connection from a client to some server, is it the case that they're starting a new TCP connection each time they talk to the backend?
A: Usually, no! When a typical browser talks to an API, the Browser opens a TCP (+TLS if HTTPS) connection to the origin. It sends one or more HTTP requests over that connection. The connection stays open for a while and is reused. If it sits idle long enough, the browser, proxy, load balancer, or server closes it. Later requests may open a new connection.

`HTTP keep-alive` is the thing that lets multiple TCP request reuse the same TCP connection; since [[HTTP 1.1]] it's been the normal default.