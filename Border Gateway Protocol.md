---
aliases:
  - BGP
---
A routing protocol used by routers to tell each other which IP prefixes they can reach, and what path traffic should take to get there. Routers use these advertisements to choose where to forward traffic.

In short: `BGP is how networks announce reachability for IP address ranges`

Example:
- Router A: I can reach `203.0.113.0/24`
- Router B: I can reach `198.51.100.0/24`

Other routers uses these announcements to decide where to send packets.

On the public internet, BGP is the major protocol that actually connects different networks, such as ISPs, cloud providers, CDNs, and large companies.