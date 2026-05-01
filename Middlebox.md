---
aliases:
  - Middleboxes
---
A middlebox is any network device that sits between the client and server, and does something to the traffic beyond just forwarding packets.

The name is deliberately vague: It's the umbrella term for "all the stuff in the middle of the internet that isn't a plain [[Router]]".
- A pure router looks at the destination IP and forwards the packet.
- A middlebox ==inspects, modifies, terminates, or filters traffic based on higher-layer information== ([[Transport Control Protocol|TCP]] state, [[Transport Layer Security|TLS]] metadata, [[HTTP]] headers, etc.)

==Middleboxes== are the reason why deploying a new [[Transport Layer]] protocol on the internet is nearly impossible. This is called "Protocol Ossification"
- They're not just forwarding, they assume things about the traffic they see, and they break anything that doesn't fit those examples (e.g. something that isn't TCP or UDP; this is why [[QUIC]], which sort of combines [[Transport Control Protocol|TCP]] and [[Transport Layer Security|TLS]], actually chose to ride on [[User Datagram Protocol|UDP]]).
	- QUIC's design goal was to prevent middleboxes from ossifying the protocol (after Google watched TCP bet frozen by middlebox assumptions); by encrypting the transport layer itself (packet numbers, ack info, most of the header), middleboxes can't see enough to "help," and so they have to treat QUIC as opaque UDP packets.

### Common Middlebox Types
- [[Network Address Translation|NAT]]: Rewrites source IP/port so that many devices behind one public IP can share it. Your home router is a NAT. Tracks TCP connection state to map replies back to the right device.
- [[Firewall]]s: Drop traffic based on rules (port, IP, protocol). Stateful firewalls track connection state.
- [[Load Balancing|Load Balancer]]s: Distribute incoming connections across backend servers ([[HAProxy]], [[Envoy]], [[Amazon Application Load Balancer|ALB]]/[[Amazon Network Load Balancer|NLB]])
- [[Reverse Proxy|Reverse Proxies]]: Terminate the client connection, make a fresh one to the backend ([[NGINX]], Cloudflare, Fastly). [[Content Delivery Network|CDN]]s are giant reverse proxies with caching.
- [[Forward Proxy|Forward Proxies]]: Corporate egress gateways that all employee traffic flows through. Often inspect or filter.
- [[Transport Layer Security|TLS]]-terminating proxies: Decrypt TLS, inspect the plaintext, re-encrypt to the backend. Common in load balancers and corporate "TLS inspection" boxes.
- Deep Packet Inspection: Look inside packets to classify or block traffic.
- Traffic shapers/WAN optimizers: Compress/dedupe/priortiize/rate-limit
- Intrusion detection/prevention (IDS/IPS): Sniff for attack patterns
- [[Web Application Firewall]]s (WAFs): Inspect HTTP for SQLi, XSS, etc.
- Carrier-grade NAT (CGNAT): ISP-levle NAT, common on mobile networks.
