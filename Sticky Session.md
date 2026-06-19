---
aliases:
  - Session Affinity
---
A [[Load Balancing]] routing behavior where requests from the same client are consistently routed to the same backend server for some period of time.

Some applications store user/session state locally on one server; we want successive requests to be routed to that server. The load balancer needs to either remember or reliably calculate that "this client" needs to go to "that backend server." Especially important for things like [[WebSockets|WebSocket]]s and other long-lived connections.

If the server that we're consistently routing to dies, the client must be sent somewhere else, and any state stored only that dead server may be effectively lost.

Note: Sticky sessions are usually not “requested” by the client or application per request. They are usually a load balancer configuration on a listener, route, service, pool, or target group.

With Sticky Sessions:
```
First request:
Client A -> Load Balancer -> Server 2

Later requests from same client:
Client A -> Load Balancer -> Server 2
Client A -> Load Balancer -> Server 2
Client A -> Load Balancer -> Server 2
```
Without Sticky Sessions:
```
Client A -> Server 1
Client A -> Server 3
Client A -> Server 2
Client A -> Server 1
```

# Common Mechanisms
| Mechanism                   | How it works                                                          | Main weakness                                                               |
| --------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Load balancer cookie        | Load balancer sets a cookie identifying the chosen backend            | Depends on browser cookies                                                  |
| Application cookie affinity | Load balancer routes based on an application cookie like `JSESSIONID` | Ties infrastructure behavior to application behavior                        |
| Source IP affinity          | Requests from the same IP address go to the same backend server       | Works poorly with Network Address Translation, mobile networks, and proxies |
| Consistent hashing          | Backend server is chosen by hashing a client, session, or user key    | Rebalancing can still move clients                                          |

Example flow:
1. Client sends first request, which doesn't have a cookie attached
2. Load balancer chooses Server 2
3. Load balancer returns the response, along with a cookie, for example: `LB_BACKEND=server-2`
4. Client includes that cookie on later requests
5. Load balancer reads the cookie and routes those requests to Server 2

```
Route: https://example.com/*
Backends:
  - app-a
  - app-b
  - app-c

Load balancing algorithm: least-connections (e.g.)
Sticky sessions: enabled
Sticky cookie name: LB_STICKY
Sticky cookie lifetime: 30 minutes
```

What's actually inside the load balancer cookie?
There are two implementation styles:
1. ==Encoded Backend Identifier== ([[Structured Token]])
	- The cookies value encodes the selected backend, often a structured token in an encoded and signed format so that the client can't read or forge "send me to backend instance `app-b`"
	- `Set-Cookie: LB_STICKY=6f8c2a9e4d...; Path=/; Secure; HttpOnly; Max-Age=1800`
```
payload = {
  backend_id: "app-b",
  expires_at: 1781459400
}

cookie = base64(payload) + "." + HMAC(secret_key, payload)
```
2. ==Lookup Key into Load Balancer State== ([[Opaque Token]])
	- `Set-Cookie: LB_STICKY=s-79d42b0c9a; Path=/; Secure; HttpOnly; Max-Age=1800`
	- The load balancer stores a mapping internally: `s-79d42b0c9a -> app-b`
		- Of course, if you have multiple load balancers (e.g. active-active), then you have to externalize this state, which can incur additional latency.
	- Later, client sends cookie with `LB_STICKY=s-79d42b0c9a`, load balancer looks it up, and sees that it routes to `app-b`. 
	- This avoids exposing backend identity to the client, but requires that the LB maintains affinity state, and potentially replicate that state across load balancer nodes.


# Comparison with a ==Shared Session Store==
Depending on the situation and what state is being stored on the server in question, a more scalable design is often:
```
Client -> Any backend -> Shared session store
```
Example:
```
Client -> Load Balancer -> server 1 -> Redis
Client -> Load Balancer -> server 3 -> Redis
Client -> Load Balancer -> server 2 -> Redis
```
Now any server can handle the request, because the session state has been centralized.
((This isn't obviously appropriate for the WebSocket case, to me))
