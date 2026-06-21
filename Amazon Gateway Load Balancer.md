---
aliases:
  - AWS Gateway Load Balancer
  - Gateway Load Balancer
---
AWS's load balancer for *network appliances,* not normal application servers.

> An [[AWS Elastic Load Balancer|Elastic Load Balancer]] type that transparently routes network traffic through a fleet of virtual appliances, such as firewalls, intrusion detection systems, intrusion prevention systems, or deep packet inspection systems.


Operates at [[Network Layer|Layer 3]], and is aware of IP packets/flows, rather than (as in Layer 4) TCP/UDP/TLS flows.


```
Internet
  |
  v
Application VPC route table
  |
  v
Gateway Load Balancer endpoint
  |
  v
Gateway Load Balancer
  |
  +--> firewall appliance A
  +--> firewall appliance B
  +--> firewall appliance C
  |
  v
Application servers
```
The application servers are usually not the direct targets. The security appliances are the targets.

