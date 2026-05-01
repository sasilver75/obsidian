
D1 is [[Cloudflare]]'s managed [[Serverless]] database with [[SQLite]]'s SQL semantics, built-in disaster recovery, and [[Cloudflare Workers]] and HTTP API access.

> "D1 allows you to create thousands of databases at no extra cost for isolation, perfect for scaling out application vibe-coding platforms. D1 pricing is based only on query and storage costs."

It runs at the edge, alongside [[Cloudflare Workers]], with data replicated globally and automatic read replication for low-latency reads near users.
- Designed for Workers: Primary access is via Workers bindings, though there's also an HTTP API.

Best for read-heavy apps already on Cloudflare's stack; less ideal for write-heavy workloads or when you need Postgres-specific features. 