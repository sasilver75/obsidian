---
aliases:
  - Load Balancer
---


In an application, we're likely to need to [[Horizontal Scaling|Horizontally Scale]] our system.

Load Balancers provide an intermediate layer that routes clients to one of the servers in its load balancing pool.

Load Balancers can operate on multiple Load Balancing Policies:
- [[Round Robin]]
- [[Consistent Hashing]]
	- IP Address vs User ID, for example; sometimes we want to route to the same server to take advantage of cached data.


You can have **Hardware Load Balancer** or **Software Load Balancers**

Commonly, we talk about **Layer 4 Load Balancers** and **Layer 7 Load Balancers**.
- TODO: Performance tradeoffs


Load Balancers shouldn't be single points of failure, they also need to be replicated!
- **[[Active-Active]]** configuration:
	- Multiple LBs running at the same time, clients randomly choose one.
	- Load balancers can register themselves with a [[Coordination Service]] so that clients know they exist.
- **[[Active-Passive]]** configuration
	- One active load balancer, the others doing nothing.
	- Other load balancers listen to coordination service to see whether active load balancer is alive, and if not, try to claim the active role for themselves.

__________

Two purposes of load balancing
- Spreads load so that you can handle more traffic than a single server can handle
- Facilitates high availability. If Server A goes down, your clients can still talk to Server B or Server A.


We can do either:
- ==Client-Side Load-Balancing==: Client is aware of all servers they can connect to: They query a registry that tells them about the existence of all of the servers. Alternatively, they query the servers themselves; if they have a reference to one server, they can ask that server about the other servers. This is how [[Redis]] does it.
	- Can be effective, because it doesn't introduce an additional hop of latency. Perhaps use for intenral traffic.[[gRPC]] supports it natively.
	- On the other hand, it requires that the clients get updates about the servers that exist.
	- Okay in situations where you don't have many clients... and when you can tolerate update delays (e.g. DNS, where... when we make a request to resolve a name to an IP address of the server, we get a list, and we're going to hit the first server on the list, and if they don't respond, we hit the second... This allows the aggregate consumers of DNS to handle the load balancing themselves.)
	- Not a great situation if I need to react rapidly to changes in the world.
- A dedicated LB appliance: 
	- When you need to be able to work with external clients that need to be able to get updates quickly...
	- Can be either software (e.g. [[HAProxy]], [[NGINX]]) or [[Hardware Load Balancer]]s, which can scale much harder. Also have cloud offerings ([[Amazon Application Load Balancer|ALB]], [[AWS Elastic Load Balancer|ELB]])
	- When applications spin up, they announce themselves to the load balancer, saying "Hey, I'd like to receive traffic."
	- The load balancer performs [[Health Check]] requests (either shallow: "Can you take a TCP connection?" to deep: "Can I look at the response from this request and see a status code?") on the servers in its LB pool. A server's health status as per the LB determines whether it receives traffic.
	- There are many algorithms in use to balance traffic.
	- For stateless services (e.g. HTTP servers), the most typical are either [[Round Robin]] or [[Random Allocation]] of requests. If we assume that requests take the same amount of time, we get approximately the even load across servers.
	- Alternatively, we can ... use [[Least Connections]] as a strategy, allocating to the server with the least ocnnections.
		- This is nice if we have a new server, and we want to make sure that the load on that serve comes up quickly to its peers' levels.
		- Can also be useful for long-running or stateful connections, like [[WebSockets|WebSocket]] connections, which might last for an hour, meaning we can have really uneven load across our servers.


![[Pasted image 20260605171626.png]]

![[Pasted image 20260605171832.png]]




Another consideration is the level at which the load balancer is operating at:

- [[Layer 4 Load Balancer]], operating at the TCP level
	- Initaites 
- [[Layer 7 Load Balancer]], operating at the HTTP level



[[Layer 4 Load Balancer]]
![[Pasted image 20260605172233.png]]
- Initiates a Layer 4 Connection for each inbound connection it receives
	- Client creates TCP connection to LB
	- LB creates parallel TCP connection to server
	- We can "kind of" pretend that the client has a direct TCP connection with the server itself.
	- Very high performance; The L4 load balancer doesn't need to do a lot of thinking; it doesn't look at packets. It just pushes packets to the *origin* (the server it's load balancing to).


[[Layer 7 Load Balancer]] (e.g. [[Amazon Application Load Balancer|ALB]])
![[Pasted image 20260605173857.png]]
- Instead of only accepting TCP connections...it actually is going to accept (say) HTTP requests! It accepts requests at the last layer, [[Application Layer|Layer 7]]. 
- It's then going to choose an arbitrary server based on its [[Load Balancing Strategy]] that it will send that request to.
- For the Server, it's not as straightforward whether the connection it has with the LB is the same one that it has with the client.
	- ==The L7  LB might have only a few connections to the servers, and many connections to clients!==
- Can be a little more expensive; they need to be able to handle application-layer responses (full HTTP requests), but if I had a single TCP connection from client to LB and multiple HTTP requests that happened over it, I can fan out those requests to all of the servers, and don't have to send them all t to the same server!
- ==Not appropriate==  in cases where I have a *connection-oriented protocol* like [[WebSockets]], for instance. For the most part, L7 load balancers aren't appropriate in those cases, and we should use L4.
- In cases where we can accept a performance hit, a L7 load balancer provides a lot of functionality, routing options, the ability to do [[TLS Termination]]... whatever we want to do, we can do it with a L7 LB; ==it's our default that we recommend in most instances, unless you're using a connection-oriented protocol like WebSockets==.


