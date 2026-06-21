---
aliases:
  - AWS Route 53
  - Route 53
---
AWS's managed [[Domain Name Service|DNS]] service. 

Use it when you want AWS-native control over domain DNDS, especially when the target is AWS infrastructure such as [[Amazon CloudFront]], an [[Amazon Application Load Balancer|AWS Application Load Balancer]], a [[Amazon Network Load Balancer|AWS Network Load Balancer]], an [[Amazon Network Load Balancer|AWS Network Load Balancer]], [[Amazon API Gateway|AWS API Gateway]], [[Amazon S3|S3]] website endpoint, or private services inside a [[Virtual Private Cloud|VPC]].


```
example.com -> where should the client connect?
```

### What Route 53 does:
- Domain Registration: You can buy/register domains through AWS
- Public Authoritative DNS: Route public names like `example.com` or `api.example.com`
- Private DNS: Route internal names inside one or more VPCs
- [[Health-Checked DNS]]s: Detect unhealthy endpoints and optionally route around them
- Routing Policies: Choose DNS answers by simple, weighted, latency, failover, geolocation, geoproximity, ([[GeoDNS]]), IP-based, or multivalue policies.
- VPC Resolver: Resolve DNS inside VPCs and between AWS and on-prem networks.

### How it Works
1. You register or already own a domain, such as `example.com`
2. You create R53 public hosted zone for `example.com`
3. R53 gives you [[Authoritative Nameserver]]s
4. Your [[Registrar]] delegates `example.com` to those Route53 nameservers.
5. You create DNS records in the hosted zone (e.g. [[A Record]]s, [[CNAME Record]]s, etc.)
6. When a user visits `example.com`, the user's [[Recursive Resolver]] eventually asks R53's authoritative nameservers for the answer.
7. Route 53 returns the DNS answer based on the record and routing policy.
8. The client connects to the returned destination.


A very common AWS production setup is:
Route 53
  -> [[Amazon CloudFront|CloudFront]]
      -> [[Amazon Application Load Balancer|AWS Application Load Balancer]]
          -> [[Amazon Elastic Container Service|ECS]] / [[Amazon EC2|EC2]]/ [[Amazon Lambda|Lambda]] / [[Kubernetes]] service

