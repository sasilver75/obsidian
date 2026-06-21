---
aliases:
  - Elastic Load Balancer
  - AWS ELB
  - ELB
---
A managed AWS load balancer created through Elastic Load Balancing service.
- An AWS-managed entry point that receives client traffic and distributes across multiple healthy backend targets, such as EC2 instances, containers, IP addresses, Lambda functions, or appliances, depending on the LB type.
- Gives clients one stable place to connect, while AWS handles distributing traffic, health-checking targets, and scaling the LB infrastructure.

Think of Elastic Load Balancing (ELB) as the AWS load balancing service/family! It contains specific load balancer types:
- [[Amazon Application Load Balancer|AWS Application Load Balancer]]: [[Application Layer|Layer 7]] for HTTP/HTTPS/gRPC
- [[Amazon Network Load Balancer|AWS Network Load Balancer]]: [[Transport Layer|Layer 4]] for TCP/UDP/TLS
- [[Amazon Gateway Load Balancer|AWS Gateway Load Balancer]]: Appliance/firewall-style traffic
- AWS Classic Load Balancer





