---
aliases:
  - Worker
---


A [[Serverless]] compute platform that runs your code on [[Cloudflare]]'s global edge network in 300+ data centers worldwide, close to users.

Key Characteristic:
- V8 isolates, rather than containers: Instead of spinning up a Node.js process per request, Workers run JS/TS or [[Web Assembly|WASM]] in lightweight V8 isolates. 
	- ==[[Cold Start]]s are only ~5ms==, vs hundreds of ms for traditional serverless.
	- Runs at the edge: Code executes at a data center closest to the user, reducing latency.
	- Web-standard APIs: Uses `fetch`, `Request`, `Response`, `Streams`, etc.
	- Bindings: Declarative connections to other Cloudflare services: [[Cloudflare D1]], [[Cloudflare Workers KV]], [[Cloudflare R2]], [[Cloudflare Durable Objects]], [[Cloudflare Queues]], AI models.
	- Limits: 10-30s CPU time dependign on plan, 128MB memory.

Common use cases:
- API routes
- Middleware/auth
- A/B testing
- Image transformation
- AI inference
- Full applications via frameworks via frameworks like [[Next.js]] (with adapter)