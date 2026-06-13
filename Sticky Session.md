---
aliases:
  - Session Affinity
---

A [[Load Balancing]] routing behavior where requests from the same client are consistently routed to the same backend server for some period of time.

Some applications store user/session state locally on one server; we want successive requests to be routed to that server. The load balancer needs to either remember or reliably calculate that "this client" needs to go to "that backend server." Especially important for things like [[WebSockets|WebSocket]]s and other long-lived connections.

If the server that we're consistently routing to dies, the client must be sent somewhere else, and any state stored only that dead server may be effectively lost.

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
| Mechanism | How it works | Main weakness |
|---|---|---|
| Load balancer cookie | Load balancer sets a cookie identifying the chosen backend | Depends on browser cookies |
| Application cookie affinity | Load balancer routes based on an application cookie like `JSESSIONID` | Ties infrastructure behavior to application behavior |
| Source IP affinity | Requests from the same IP address go to the same backend server | Works poorly with Network Address Translation, mobile networks, and proxies |
| Consistent hashing | Backend server is chosen by hashing a client, session, or user key | Rebalancing can still move clients |

Example flow:
1. Client sends first request
2. Load balancer chooses Server 2
3. Load balancer returns a cookie, for example: `LB_BACKEND=server-2`
4. Client includes that cookie on later requests
5. Load balancer routes those requests to Server 2


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

