
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





