---
aliases:
  - Load Balancer
---
In an application, we're likely to need to [[Horizontal Scaling|Horizontally Scale]] our system.

Load Balancers provide an intermediate layer that routes clients to one of the servers in its load balancing pool.

Load Balancers can operate on multiple Load Balancing Policies: See [[Load Balancing Strategy]]
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


_____________
# Important Question

Q: Imagine that we have a microservices architecture of some sort. Do we typically have a load balancer in front of every pool of service instances?

A: Every replicated service needs *some sort of load balancing mechanism,* but no, that doesn't always mean a separte load balancer appliance in front of every service pool.

A common shape is:
```
Internet
	-> Edge Load Balancer / API Gateway / Ingres
	-> Service A instances
	-> Service B instances
	-> Service C insteances
```
For internal service-to-service calls, the balancing is often handled by one of these:
- [[Kubernetes]] Service: `orders` calls `payments.default.svc.cluster.local`, and K8s spreads traffic across payments Pods.
- [[Service Mesh]]: Evoy/Linkerd/Istio proxy sidecars balance traffic between instances for your application instances
- Client-Side discovery/routing: The caller fetches healthy instances from Consul/Eureka/etc and chooses one
- Internal Load Balancer: Common in cloud or VM-based systems, especially without K8s/mesh
- [[Domain Name Service|DNS]]-based balancing: Simpler, but weaker health/routing

Mental model:
> "Each service exposes some sort of stable logical address (not necessarily [[Internet Protocol|IP]] address) backed by a real load balancer. Sometimes it is a virtual service, sidecar proxy, DNS name, or client-side discovery layer."

So when orders service calls: `http://payments`, `payments` is stable, but the actual instances behind it (`payments-1: 10.0.2.14`, `payments-2: 10.0.3.91`, etc.) are not; they can be added, removed, replaced. Callers shouldn't need to know or care.

Common forms:
- Kubernetes Service: `ordres -> payments.default.svc.cluster.local -> payment pods`
- Service Mesh: `orders -> payments -> local Envoy sidecar -> healthy payment instance`
- Cloud/VM Setup: `orders -> internal-paymentes-lb.example.com -> payment VMs`
- Client-Side Discovery: orders asks [[HashiCorp Consul|Consul]]/Eureka for "payments" instances, then chooses one.

For public traffic, you typically have a small number of edge load balancers/API gateways routing to many services. For internal traffic, you often have many logical load-balanced service endpoints, but not necessarily many separately managed load balancer boxes.


______________

Q: How do Load Balancers not become a dangerous single point of failure?

A:

Suppose we have a setup like this:
```
clients
	-> Load balancer
		-> payments-1
		-> payments-2
		-> payments-3
```
This load balancer enables clients to not need to know about every payments instance: they just call one stable address, and the load balancer distributes traffic.

But now there's a new problem: The LB itself is a [[Single Point of Failure]]! If it fails, then clients cannot reach the service instances, even if the instances themselves are healthy!

For the service to stay reachable, we need redundancy at every required step in the request path!

> So how do clients keep using one stable service address while the load balancer itself is made redundant?

There are a number of solutions, here:[]()
1. [[Active-Passive]] Load Balancers
2. [[Active-Active]] Load Balancers
3. A Managed Load Balancer

They all work the same way:
> Make the load-balancing layer redundant while preserving one stable address for clients!

### [[Active-Passive]] Load Balancers
We run ==two load balancers==
```
clients
	-> shared Virtual IP
		-> active LB
		-> payments instances
		
		-> passive LB
```
In this case, only one load balancer actively handles traffic, while the other waits.
If the active one dies:
```
clients
	-> same shared virtual IP
		-> passive LB becomes active
		-> payments instances
```
The clients keep using the same address, and behind the scenes, ownership of the address moves to the standby load balancer.

This is often implemented with a ==[[Virtual IP Address]]== and a failover protocol like [[Virtual Router Redundancy Protocol|VRRP]]/Keepalived
- A Virtual IP Address is a stable IP address that is not permanently tied to one physical machine. Instead, it can "move" between machines.

### [[Active-Active]] Load Balancers
In this scheme, we run ==multiple load balancers==, and all of them serve traffic at the same time
```
clients
	-> stable frontend address
		-> LB A
		-> LB C
		-> LB C
			-> payments instances
```
If one LB fails, traffic continues through the remaining load balancers.

Implementations:
1. DNS returns multiple load balancer addresses
```
payments.example.com -> 10.0.0.11
payments.example.com -> 10.0.0.12
payments.example.com -> 10.0.0.13
```
Some clients connect to LB A, some to LB B, etc.
If B dies, [[Health-Checked DNS]] stops returning 10.0.0.12.  This is simple, but failover can be imperfect because clients/resolvers cache DNS records.

2. One Virtual IP, multiple load balancers behind network routing
Clients call one IP:
```
payments.example.com -> 10.0.0.50
```
Multiple load balancers can serve that IP:
```
10.0.0.50
	-> LB A
	-> LB B
	-> LB C
```
Network routers distribute flows across the LBs, often using [[Equal-Cost Multi-Path]] (ECMP) Routing
If LB B fails, it stops advertising the route, or health checks remove it, and new traffic goes to A and C.

3. [[Anycast]]
The same IP is announced from multiple places:
```
203.0.113.10 announced by region A
203.0.113.10 announced by region B
203.0.113.10 announced by region C
```
The network routes clients to a nearby/available location. If one location fails, its route is withdrawn and traffic shifts elsewhere. This is common for global edge systems, CDNs, DNS providers, large platforms.

### A Managed Load Balancer
In AWS/GCP/Azure, when you create a "Load Balancer" (e.g. [[Amazon Application Load Balancer|ALB]]), you are usually not getting a single VM, you're getting a managed service that's internally redundant:
```
clients
	-> cloud LB DNS name / IP
		-> provider-managed LB nodes across zones
			-> your service instances
```
The cloud provider handles the active-active or failover mechanics for the load-balancing tier.

You still have to configure it correctly, for example:
- Use multiple availability zones
- Register healthy backend instances in multiple zones
- Configure health checks



So we know that we have teh following High-Availability modes:
1. Active-Passive: One serves traffic, one waits
2. Active-Active: Multiple serve traffic at the same time

And then we have different traffic steering mechanisms that describe how clients/traffic reach the available LB noes:
3. Virtual IP/routing/ECMP
4. DNS failover or DNS round-robin
5. Anycast

And combinations of these are common:
- Self-managed HAXProxy pair = active-passive + virtual IP
- Self-managed HAProxy cluster = active-active + DNS round-robin or ECMP
- Cloud load balancer = managed LB + usually active-active internally
- Multi-region service = managed regional LBs + DNS failover
- Global CDN/Edge = active-active + anycast

> First, decide whether the LB layer is [[Active-Passive]] or [[Active-Active]], and then decide how traffic finds the healthy LB nodes: whether by [[Virtual IP Address|VIP]], [[Domain Name Service|DNS]], [[Equal-Cost Multi-Path|ECMP]], [[Anycast]], or some provider-managed infrastructure.


There's even more to say on this: [[Service Mesh]]es, [[Kubernetes]] Services, [[Client-Side Load Balancing]], [[Border Gateway Protocol|BGP]] routing, appliance clusters, and cloud provider internals add more detail/complexity.