---
aliases:
  - SSL Termination
---

When a device or service decrypts [[HTTPS]]/[[Transport Layer Security|TLS]] traffic.

```
Client
	-> HTTP/TLS
Load Balancer / Proxy decrypts results
	-> HTTP or new HTTPS/TLS
App Server
```
The TLS session from the client ends at the intermediary. 

This commonly happens at:
- The [[Content Delivery Network|CDN]] [[Edge Server|Edge]], like [[Cloudflare]] or [[Amazon CloudFront|CloudFront]]
- [[Load Balancing|Load Balancer]], like [[Amazon Application Load Balancer|AWS ALB]]
- [[Reverse Proxy]], like [[NGINX]] or [[Envoy]]
- [[API Gateway]]
- [[Kubernetes]] Ingress controller
- Service Mesh Sidecar

# Why terminate TLS before the App?
- Benefits
	- Centralized certificate management
	- [[Web Application Firewall|WAF]]/security inspection
	- HTTP routing by host/path/header (e.g. for [[Application Layer|Layer 7]] LB)
	- Offloads TLS work from app servers
	- Enables CDN caching/compression
	- Simplifies app server config
- Costs
	- Intermediary service/device can see plaintext traffic
	- Need to protect internal network or re-encrypt
	- Need correct forwarding headers like `X-Forwarded-Proto`
	- App must know original client scheme/IP if needed


Important Headers
- After termination, proxies often add:
```
X-Forwarded-For: original client IP
X-Fordwarded-Proto: https
X-Forwarded-Host: example.com
Forwarded: for=...; proto=https: host=...
```
Apps need these for redirects, [[Logging]], [[Rate Limiting]], Auth callback URLS, security decisions.


______________

TLS Termination means the encrypted HTTPS connection ends at the intermediary.

`client -> encrypted TLS -> API gateway / load balancer`
`API gateway decrypts HTTP`
`API gateway -> backend`

Once TLS is terminated, the gateway can see the HTTP request:
```
method: GET
path: /v1/orders
headers
cookies
body
```
Then may it forward to the backend either:
- plaintext HTTP
- a new HTTPS/TLS connection

So common patterns are:
- TLS termination at the edge, plaintext internally
- TLS termination at the edge, re-encrypted to backend
- TLS passes through the edge, terminates at backend
- [[Mutual TLS|mTLS]] between proxies/services internally

Why it matters:
- TLS termination is important because many gateway features require decrypted HTTP traffic
```
route by path/header
authenticate requests
rate limit by user/API key
apply WAF rules
inspect cookies/JWTs
log HTTP metadata
compress responses
cache responses
rewrite headers
```


If TLS is NOT terminated and is just passed through, the intermediary mostly sees: source IP, destination IP, port, maybe SNI hostname, packet/connection metadata. It can't inspect `/api/users` vs `/api/payments` because that's inside encrypted HTTP.
- So both [[API Gateway]]s and [[Layer 7 Load Balancer]]s both often do [[TLS Termination]] (and [[TCP Termination]])
