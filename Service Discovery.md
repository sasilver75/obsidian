

The mechanism by which ==one service finds the network location of another service at runtime.== 

In distributed systems, service instances are constantly changing. Machines die, containers restart, [[Autoscaling]] adds/removes replicas, deployment roll forward, regions fail over, etc. You can't just rely on static IP addresses!

A typical flow:
1. A Payments Service instance starts
2. This `payments-service` instance is registered in Service Discovery, either by the service itself, an orchestrator, or by a sidecar.
3. Service Discovery tracks whether the instance is healthy
4. Later, `order-service` or its proxy resolves `payments-service`, usually on service startup, via [[Domain Name Service|DNS]], via periodic refresh, or via a watch stream.
5. `order-sevice` caches this routing information
6. On each actual request, `order-service` uses the cached endpoint, a local proxy, or a load balancer address.

So the request path, at runtime, from `order-service` to `payments-service`, is usually:
- `order-serivce -> payments-service`
- OR `order-service -> load balancer -> payments-service`


# Common Patterns
- Startup Lookup: Client resolves services once, caches result
- Period Refresh: Client refreshes endpoint list every N seconds
- Watch/subscribe: Registry pushes endpoint changes to clients/proxies
- [[Domain Name Service|DNS]] resolution: Client resolves a service name, with DNS caching/TTL
- Proxy/LB/[[Service Mesh]]: App calls a stable name/local proxy; proxy keeps endpoint list fresh


# Server-Side Discovery
- When the caller sends requests to a stable endpoint, and some server-side component chooses the actual service instance.
- In this case, the client themselves does *not* need to know every healthy backend instance.
If `orders-service` -> `payments.internal`
- `payments.interal` points to a [[Load Balancing|Load Balancer]], which knows the healthy `payments-service` instances, then it forwards the request to one of them.
```
payments-service instances:
10.0.0.11:8080
10.0.0.12:8080
10.0.0.13:8080

orders-service -> load balancer -> 10.0.0.12:8080
```
Flow:
1. `payments-service` instance start
2. Instances are registered with discovery by the orchestrator, service itself, or a health-check system
3. A Load Balancer keeps track of healthy instances with [[Health Check]]s
	- (Load Balancer itself gets this list from a control plane of some sort)
	- > "The LB's backend pool is maintained by a control plane, which may watch K8s, Consul, ECS target groups, DNS or another registry. Some modern proxies can watch discovery sources directly, but the key point is that backend membership is refreshed out of band, not looked up on every request."
4. `orders-service` calls a stable name/address (e.g. `payments.interal`)
5. The Load Balancer/Proxy on the other end routes the request to one healthy instance.
Pros:
- Simpler clients
- Centralized load balancing/routing
- Easier to support many client languages
- Traffic policy can live in one place
Cons:
- Extra network hop
- LB becomes critical infrastructure
- Less per-client routing control
Examples:
- [[Kubernetes]] Services
- [[Amazon Application Load Balancer|ALB]]/[[AWS Elastic Load Balancer|ELB]]
- Internal [[Domain Name Service|DNS]] pointing to a load balancer
- [[Envoy]]/[[HAProxy]] fronting a service pool

# Client-Side Discovery
- The calling service is responsible for finding and choosing the target service instance.
- Instead of calling a stable load balancer and letting it pic an instance, the client gets a list of healthy examples and selects one itself.
```
payments-service instances:
10.0.0.11:8080
10.0.0.12:8080
10.0.0.13:8080
```
`orders-service` might keep this list locally, and choose one when it needs to call payments.
Flow:
1. `payments-service` instances are registered in a service registry
2. `orders-service` resolves `payments-service`, usually on startup, via periodic refresh, or via a watch stream.
3. `orders-service` caches this list of healthy payment instances
4. When making a request, `orders-service` chooses one instance using [[Round Robin]], [[Random Allocation]], via [[Consistent Hashing]], etc.
5. `orders-service`calls that instance directly.
Pros
- Fewer network hops
- Client can make smarter routing decisions
- No central load balancer needed for each service call
Cons:
- Client code/libraries are more complex
- Each language needs compatible discovery logic
- Bad clients can route incorrectly if their cache is stale
- Harder to centralize traffic policy
Examples
- Netflix Eureka + client libraries
- [[HashiCorp Consul|Consul]] service discovery with client-side load balancing



