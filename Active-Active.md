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

Mechanisms:

[[Domain Name Service|DNS]]-based:
```
payments.example.com -> 10.0.0.11
payments.example.com -> 10.0.0.12
payments.example.com -> 10.0.0.13
```
Different clients get or choose different IPs

Routing-based/[[Equal-Cost Multi-Path|ECMP]]
```
payments.example.com -> 10.0.0.50
```
Multiple load balancers advertise a route for 10.0.0.50, and network routers choose one per connection/flow.

[[Anycast]]-based
```
203.0.113.10 is announced from multiple locations
```
Internet routing sends the client to one available/nearby location


The important parts are:
- Multiple nodes are live, concurrently.
- Traffic is shared across them during normal operation.
- Failure of one node reduces capacity, but should not make the service unavailable.
- Some mechanism must steer traffic among them, such as [[Domain Name Service|DNS]], [[Equal-Cost Multi-Path|ECMP]]/Routing (e.g. [[Border Gateway Protocol|BGP]]), [[Anycast]], or cloud-provider infrastructure.





