
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






