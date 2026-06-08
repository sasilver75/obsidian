See: [[Domain Name Service|DNS]]

Health-Checked DNS means that your DNS provider or traffic-management system actively checks the targets and changes DNS answer based on their health.

If we had
```
payments.example.com
	-> 203.0.113.10 healthy
	-> 203.0.113.20 unhealthy
```

A health-checked DNS system may stop returning the healthy IP.

This is something that you would configure in a provider/tool, for example:
- [[Amazon Route 53|AWS Route 53]] health checks
- [[Cloudflare]] Load Balancing
- [[Google Cloud DNS]] routing/health checks via [[Google Cloud Load Balancing|Cloud Load Balancing]] integrations
- Azure Traffic Manager
- NS1/Akamai/other DNS traffic managers
- [[HashiCorp Consul|Consul]] DNS for internal service discovery

So the DNS protocol is still DNS, but the [[Authoritative Nameserver]] is backed by health-check logic.

==IMPORTANT LIMITATION:==
- DNS failover is not instant or perfectly reliable, because both clients and resolvers cache answers according to [[Time to Live|TTL]]s, and some ignore low TTLs. This means that failover can be imperfect/slow.