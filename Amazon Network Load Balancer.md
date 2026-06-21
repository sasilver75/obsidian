---
aliases:
  - AWS Network Load Balancer
  - AWS NLB
  - NLB
---
AWS's managed [[Transport Layer|Layer 4]] load balancer used in as a LB type in [[AWS Elastic Load Balancer]]. It's not HTTP-aware in the way that [[Amazon Application Load Balancer|AWS Application Load Balancer]] is.
- Once NLB chooses a target for a TCP connection, that connection stays with that target for the life of the connection. This matters for HTTP keep-alive, [[WebSockets|WebSocket]]s, databases, game servers, MQTT, SMTP, custom protocols, and other long-lived connections.
- ==NLB routes connections or flows, not individual HTTP requests.==

### How NLB Works
- A NLB has listeners, target groups, and targets
	- A listener accepts traffic on protocol and port, e.g. TCP 443, TLS 443, UDP 53, or TCP 5432.
	- A target group contains backend targets, like EC2 instances, private IP addresses, or an ALB.
- For TCP traffic, NLB chooses a target using flow information such as source IP address, source port, destination IP address, destination port, protocol, and TCP sequence information.
- For UDP traffic, it uses the flow tuple so packets from the same flow keep going to the same target.

# Comparison with ALB
NLB asks:
> "What TCP/UDP/TLS flow is this? What healthy target should own this connection or flow?"
- Use for raw TCP/UDP services, static IP allowlisting, TLS pass-through very high connection scale, low latency, source IP preservation, PrivateLink endpoint services, or long-lived connections.

ALB, in contrast, asks:
> "What HTTPS request is this? What host/path/header/method should decide the backend?"
- Use for a normal web application or API.