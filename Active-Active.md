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
Different clients get or choose different IPs (basically client-side load balancing for your load balancer)
- ((Seems to me to be the best option))

Routing-based/[[Equal-Cost Multi-Path|ECMP]]
```
payments.example.com -> 10.0.0.50
```
Multiple load balancers advertise a route for 10.0.0.50, and network routers choose one per connection/flow.
- ((Not sure I understand this option yet))

[[Anycast]]-based
```
203.0.113.10 is announced from multiple locations
```
Internet routing sends the client to one available/nearby location
- ((How do we stop duplicate work, in the case where we're just anycasting to a bunch of people? Is Anycast not just Broadcast where we only listen to the first answer? This doesn't seem to help write througput))


The important parts are:
- Multiple nodes are live, concurrently.
- Traffic is shared across them during normal operation.
- Failure of one node reduces capacity, but should not make the service unavailable.
- Some mechanism must steer traffic among them, such as [[Domain Name Service|DNS]], [[Equal-Cost Multi-Path|ECMP]]/Routing (e.g. [[Border Gateway Protocol|BGP]]), [[Anycast]], or cloud-provider infrastructure.





