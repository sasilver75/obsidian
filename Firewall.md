---
aliases:
---
Types:
- "Firewall" ([[Network Firewall]]): Works at the [[Network Layer|L3]] and [[Transport Layer|L4]] (Network, Transport) layers.
	- Allow/Deny decisions are based on IP addresses, ports, and protocols
	- Stops port scans, direct attacks on services that shouldn't be exposed, volumetric DDoS
	- ==Typically what someone means when they say "Firewall"==
- "WAF" ([[Web Application Firewall]]): Works the [[Application Layer|L7]] (Application/HTTP) layer.
	- Allow/Deny decisions are based on HTTP method, URL, headers, body content
	- Stops SQL injection, XSS, RCE attempts, credential stuffing and brute-force login attacks, Application-layer DDoS, Bots scarping content, sensitive data exfiltration patterns in responses.
	- Can't stop network-layer attacks, because it doesn't see them. Cannot stop anything inside an encrypted body that it can't decrypt.
- (Uncommon, marketing): [[Next-Generation Firewall]]

> *"Network firewalls are the bouncer at the door. They check IDs, and only let through people that are on the list. Web Application Firewalls (WAFs) are the security guard inside. They watch what people do once they're in, and stop people being naughty.*
> The bouncer can't tell who's a shoplifter, and the inside guard can't stop a flood of people from breaking down the door.

How they often work together for a hardened web application
```
Internet
     ↓
  DDoS scrubbing  ← stops volumetric attacks (L3/L4 floods)
     ↓
  Network firewall  ← only 80/443 open to the world
     ↓
  Load balancer (terminates TLS)
     ↓
  WAF  ← inspects decrypted HTTP, blocks SQLi/XSS/etc.
     ↓
  Application
```


## Modern Blurring
- Some Cloud [[Web Application Firewall|WAF]]s bundle DDoS protection, bot arrangement, rate limiting, and geo-blocking, which are features that overlap with traditional firewalls and load balancers.
- [[Next-Generation Firewall]]s (NGFWs) like Palo Alto, Fortinet can do [[Application Layer|L7]] inspection themselves, they can decrypt TLS and inspect HTTP. Conceptually they're [[Network Firewall]]s + [[Web Application Firewall|WAF]]s + IDS (Intrusion Detection Systems) in a box.
- [[API Gateway]]s ([[Kong]], [[Apigee (GCP Managed Gateway)]])  do WAF-adjacent things (auth, rate limiting, schema validation) for API traffic specifically.
- [[Service Mesh]]es ([[Istio]], [[Linkerd]]) also push some of this: [[Mutual TLS|mTLS]], [[Rate Limiting]], Policy, down into the cluster.



