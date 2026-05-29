
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

