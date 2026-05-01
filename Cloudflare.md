An internet infrastructure company that started as a a [[Content Delivery Network|CDN]] and DDoS protection service, and has since expanded into a broad edge cloud platform.
- They built a massive global network originally for CDN/security, and now they're layering compute and storage on top of it.
	- Global by default (no region selection)
	- No Egress fees on R2
	- Edge-first compute via workers

Core product areas:
- ==[[Content Delivery Network|CDN]] and performance==: Caches static assets at 300+ global data centers; the original product!
- ==Security==: DDoS mitigation, [[Web Application Firewall|WAF]], bot management, zero trust / SASE, email security
- ==[[Domain Name Service|DNS]]==: Runs `1.1.1.1`, the public DNS resolver; also authoritative DNS for millions of domains.
- ==Developer platform:==  
	- [[Cloudflare Workers]]: Serverless JS/TS/WASM at the edge with tiny 5ms cold start
	- [[Cloudflare D1]]: Serverless SQLite database with global replication, used by Workers
	- [[Cloudflare Workers KV]]:  Eventually-consistent global KV store for high reads
	- [[Cloudflare R2]]:  S3-compatible object storage w/ zero egress fees
	- [[Cloudflare Durable Objects]]: Single-instance stateful actors w/ strong consistency for counters, locks, game state, chat rooms
	- [[Cloudflare Queues]]: Managed message queues with [[At Least Once]] delivery
	- [[Cloudflare Workers AI]]: Serverless GPU inference for open models, called by Workers
	- [[Cloudflare Vectorize]]: Managed vector database for embeddings/similarity search, built for RAG pipelines alongside Workers AI
- ==Network services==: Magic Transit, Magic WAN, Argo smart routing










