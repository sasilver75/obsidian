
A place where network connection stops being one continuous connection, and is ended/handled by an intermediary, usually a [[Load Balancing|Load Balancer]], [[Proxy]], [[API Gateway]], Ingress controller, or [[Content Delivery Network|CDN]].


TCP termination means that a device/service accepts and ends the client's TCP connection, and then usually opens a separate TCP connection to the backend.
```
Client
	-> TCP connection #1 ->
Load Balancer/ Proxy
	-> TCP connection #2 ->
App Server
```
The client doesn't maintain a TCP connection to the app server, it's connected to the intermediary.


_______

Termination means "this layer of the connection ends here."

In infra diagrams, it usually means a proxy/load balancer/API becomes the endpoint for that protocol, and then starts a separate connection upstream.

`client -> TCP connection -> load balancer`
`load balancer -> separate TCP connection -> backend service`

The backend is not on the same TCP connection as the client. The load balancer receives bytes from the client, then forwards them over another connection.

To be clear:
```
Instead of:
Browser <---------------- TCP Session ----------------> App Server Instance

We have:
Browser <--------TCP Session A----> Load Balancer <------TCP Session B------> App Server Instance
```
The load balancer terminates the browser's TCP connection, because, from the browser's perspective, the LB is the server it is connected to.
It then opens or reuses a separate TCP connection to the application.

Terminate doesn't mean "kill the connection," it means "the TCP session ends here."
Why have two TCP connections?
- Once the LB terminates TCP, it can actively manage the connection:
	- Choose a healthy backend
	- Pool/reuse backend connections
	- Retry safe requests + Enforce timeouts
	- Apply backpressure
- If it also does [[TLS Termination]] and speaks HTTP, it can do even more.
	- Route by path/header
	- auth
	- rate limiting
	- WAF
	- ...

In the case of TCP termination:
- The application server instance does not directly see the client's TCP connection; it sees the *load balancer* as its peer. So if the app needs to original clip IP, the LB must pass it along somehow, e.g. `X-Forwarder-For` header.





