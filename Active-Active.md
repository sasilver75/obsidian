A High-Availability (HA) pattern where multiple redundant components are live at the same time and share production traffic.

In the [[Load Balancing|Load Balancer]] case:
```
clients
	-> traffic steering layer (mechanism that chooses/reaches one of several load balancers)
		-> LB A
		-> LB B
		-> LB C
			-> service instances
```
All of LB A/B/C are active; if one fails, traffic is sent to the remaining nodes.

Routing Mechanisms:

[[Domain Name Service|DNS]]-based:
```
payments.example.com -> 10.0.0.11
payments.example.com -> 10.0.0.12
payments.example.com -> 10.0.0.13
```
- Two options
	1. Multiple-address DNS response, and clients/OSs/browsers/recursive resolvers may choose different addresses as they wish.
	2. DNS provider returns different answers based on policies (health, geography, weight, latency). This is common with [[Health-Checked DNS]], [[GeoDNS]], etc.
- A limitation is that DNS answers are cached by clients/recursive resolvers, so clients might keep using the old IP until the cached answer's TTL expires. So DNS failover is simple and scalable, but not instant or perfectly precise.

Routing-based/[[Equal-Cost Multi-Path|ECMP]]
```
payments.example.com -> 10.0.0.50
```
Multiple load balancers advertise a route for 10.0.0.50, and network routers choose one per connection/flow.
- In ECMP, routers have multiple equal-cost next hops for the same destination. They usually choose a next hop by hashing the flow, often using the 5-tuple: source IP, destination IP, source port, destination port, and protocol.
- The goal is that all packets for one TCP connection keep going to the same load balancer, while different connections are spread across different load balancers.

[[Anycast]]-based
```
203.0.113.10 is announced from multiple locations
```
Internet routing sends the client to one available/nearby location
- The same IP prefix is announced from multiple locations, but routing selects one best path, so a given packet normally goes to one location. This means Anycast does not create duplicate work by itself. It also does not improve write throughput by itself. It only gets the request to a nearby/available ingress point.



The important parts are:
- Multiple nodes are live, concurrently.
- Traffic is shared across them during normal operation.
- Failure of one node reduces capacity, but should not make the service unavailable.
- Some mechanism must steer traffic among them, such as [[Domain Name Service|DNS]], [[Equal-Cost Multi-Path|ECMP]]/Routing (e.g. [[Border Gateway Protocol|BGP]]), [[Anycast]], or cloud-provider infrastructure.





