

# Languages and Runtimes
- Go: Concurrency primitives (goroutines, channels), fast, simple, dominant for infra/cloud-native
- Rust: Memory safety and performance; used where you'd previously reach for C++ or for low-latency services. 
	- `Tokio` is the async runtime, and `Axum`/`Actix` are the web frameworks.
- TypeScript/Node: Increasingly serious for backend, not just BFFs.
- Python: Still huge, especially in AI/data.
- ...


# Web Frameworks
- Go: stdlib `net/http`, Echo, Gin Fiber, Chi, Huma
- Rust: Axum (Tokio team's, dominant)
- Node/TS: Express (legacy default), Fastify (faster, schema-first), Elysia (Bun-native)
- Python: [[FastAPI]] (default for new APIs, async, Pydantic-typed, OpenAPI auto), Django+DRF, Flask (legacy)
- ...

# API Paradigms
- REST: Still the default
- GraphQL: Well past peak hype but stable in places where federation pays off
- [[gRPC]]: Standard for inter-service in polyglot environments. 
	- `Connect` is gaining ground.
- [[tRPC]]: Type-safe RPC for TS-only stacks. No schema language, types flow through.
- [[WebSockets|WebSocket]]: bidirectional, persistent
- [[Server-Sent Event]]s (SSE): One-way streaming over plain HTTP, the default transport for LLM streaming today.
- [[Webhook]]s: Outbound HTTP callbacks. `Svix` is the dominant managed service.
- [[Model Context Protocol]]: Standard for AI tool/agent integration, becoming a de facto API layer for AI consumers.

# Databases, Relational
- [[PostgreSQL|Postgres]]: The default, increasingly absorbs the use cases of other domains
	- JSON support
	- [[Full-Text Search Index|Full-Text Search]]
	- Vectors via [[pgvector]]
	- Queues via [[pg_cron]] + SKIP LOCKED
	- Time-Series via [[TimescaleDB]]
	- Hosted providers worth knowing about:
		- [[Neon]]: serverless, branching
		- [[Supbase]]: Postgers + auth + realtime + storage
		- Xata
		- Crunch Bridge
		- [[Railway]]
		- Fly Postgres
- [[MySQL]]: Still huge, especially via  [[PlanetScale]]/[[PlanetScale|Vitess]] for horizontal scaling
- [[SQLite]]: Server-side renaissance. Litestream (replication), LiteFS (Fly.io), Turso (libSQL fork). Great default for small/medium apps.
- [[CockroachDB]]: Postgres-wire-compatible, horizontally scalable, strong consistency. Serverless tier.
- [[YugabyteDB]]: Similar shape to Cockroach
- [[TiDB]]: MySQL compatible distributed database
- [[Google Spanner]], [[Amazon Aurora]], [[Amazon AuAmazon Aurora DSQLrora DSQL]]: Managed globally-distributed SQL

# Non-Relational
- [[Redis]]: KV cache, queues, pub/sub, locks, rate limiting
- [[Memcached]]: Pure cache, simpler than Redis
- [[Amazon DynamoDB|DynamoDB]]: AWS-managed KV/document, single-digit ms, single-table design is the pattern to learn.
- [[Apache Cassandra|Cassandra]]: Wide-column, write-heavy, eventually consistent. [[ScyllaDB]] is the C++ rewrite, much faster.
- [[MongoDB]]: Still widely deployed KV store

# Specialized DBs
- Vector databases for embeddings and [[Vector Search|Semantic Search]]
	- [[pgvector]]: Postgres expansion, usually the right answer
	- [[Pinecone]]
	- [[Weaviate]]
	- QDrant, Milvus, Chroma, [[LanceDB]] (embedded), Turbopuffer
- Search:
	- [[ElasticSearch]] (license drama, OpenSearch fork by AWS)
	- Meilisearch, Typesense, Algolia
	- [[PostgreSQL|Postgres]] built-in FTS + [[ParadeDB]]/[[pg_search]]
		- ...
- Time-series: [[TimescaleDB]], [[InfluxDB]], QuestDB, ...
- OLA/analytics: ClickHouse, [[DuckDB]], [[Apache Druid]], Pinot, StarRocks
- Data warehouse: [[Snowflake]], [[Google BigQuery|BigQuery]], [[Databricks Lakehouse]], [[Amazon Redshift|Redshift]]
- Graph: Neo4j, Dgraph, Memgraph, plus Apache AGE (Postgres extension)
- Embedded: [[SQLite]], [[DuckDB]], [[RocksDB]] (KV), LMDB


Trend: "==Just use Postgres==" is the dominant default because of `pgvector`, `JSONB`, and good FTS. The case for adding a second store has just gotten more difficult.

# Query Layer/ORMs
- TS: 
	- [[Drizzle]] (SQL-shaped, typesfe, growing fast)
	- [[Prisma (ORM)|Prisma]] (schema-first, heavier)
- Python:
	- [[SQLAlchemy]] 2.0 (now async-native)
- Go: sqlc (Genertes code from SQL, increasingly hte default)
- Rust: sqlx (compile-time-checked SQL)

# Caching Patterns
- Application Cache (in-process): LRU, ...
- Distributed Cache: [[Redis]]/Valkey/[[Memcached]]
- CDN-level Caching: Cloudflare, Fastly, Vercel
- [[Read-Through Cache]], [[Write-Through Cache]], [[Write-Back Cache]]/[[Write-Back Cache|Write-Behind Cache]], [[Cache-Aside]]
- [[Cache Stampede]]/Thundering Herd
- [[Cache Invalidation Strategy|Cache Invalidation]]

# Queues/Streaming/Event Infra
- Message Queues: [[Amazon SQS|SQS]], [[RabbitMQ]], [[Redis Streams]], [[NATS]], [[PostgreSQL|Postgres]] as a queue (pg + SKIP LOCKED + River, Graphile Worker, pgmq)
- Streaming/log-baed: 
	- [[Kafka]] is still the ehavyweight
	- [[Redpanda]] (C++ Kafka-compatible, no JVM)
	- [[Warpstream]] (Kafka over S3)
	- [[Amazon Kinesis|AWS Kinesis]]
	- [[Apache Pulsar]]
- Cloud-native: 
	- [[Google Pub/Sub]]
	- [[Amazon SNS|AWS SNS]]/[[Amazon SQS|AWS SQS]]/[[Amazon EventBridge|AWS EventBridge]], Azure Service Bus
- Patterns: 
	- [[At Least Once]] vs [[Exactly Once]]
	- [[Outbox Pattern]]
	- [[Idempotency]]
	- [[Dead Letter Queue]] (DLQ)
	- [[Backpressure]]
	- [[Fan-Out]], [[Fan-In]]


# Background jobs and workflows
- Traditional job queues: [[Celery]]
- Durable execution/workflow engines: The idea is ==code-as-workflow with checkpointing, retries, sleep-for-days==.
	- [[Temporal]]: The heavyweight, self-cost or cloud
	- Restate: Newer, lighter
	- Inngest: Serverless-native, event-driven
	- AWS Step Functions: Managed State Machines
- Schedulers: cron, Quartz, Vercel Cron, GitHub Actions schedule,...

If you don't know durable execution, learn it! It's the most consequential backend pattern of the last few years. The mental model is that you can write code as if it never crashes, and the engine persists each step.


# Auth and Identity
- Standards
	- [[OAuth]] 2.1
	- [[OpenID Connect|OIDC]] (OpenID Connect = identity on top of OAuth)
	- [[Security Assertion Markdown Language|SAML]] (enterprise [[Single Sign-On|SSO]])
	- [[Platform-Agnostic Security Token|PASETO]] ([[JSON Web Token|JWT]] alternative)
	- [[Web Authentication API|WebAuthn]]/passkeys
	- [[JSON Web Token|JWT]] still ubiquitous
- Patterns: Rotating refresh tokens, [[Proof Key for Code Exchange|PKCE]], session vs token authn, [[Mutual TLS|mTLS]] for service-to-service communication.
- Hosted:
	- [[Auth0]] (Okta)
	- Clerk
- Self-host:
	- Keyclock, Ory, ...
- Authz: [[Open Policy Agent]], Cedar (AWS)

# Observability:
- Metrics: Numerical time-series
	- [[Prometheus]] is the de-facto
- Logs: Structured events
	- [[Grafana Loki]], [[Elastic Logs]]
- Traces
	- [[Jaeger]], [[Grafana Tempo]]
- [[OpenTelemetry Protocol|OpenTelemetry]] (OTel): THE standard now. Vendor-neutral instrumentation for capturing metrics/logs/traces.
- [[Sentry]] for error monitoring, performance, sentry replay
- Concepts:
	- [[Service Level Objective]], [[Service Level Indicator]], [[Service Level Agreement]]
	- [[RED Method]]
	- [[USE Method]]
	- [[Golden Signals of Monitoring]] (Google SRE)
- Commercial [[Application Performance Monitoring]] (APM) such as [[Datadog]], Dynatrace, [[New Relic]], [[Honeycomb]], [[Grafana]], etc.


# Containers and Orchestration
- [[Docker]]: Container runtime
	- Other [[Open Container Initiative]] (OCI) alternatives such as [[Containerd]], [[Podman]], CRI-O.
- [[Kubernetes]] is still the dominant orchestrator.
	- Vocab: Pod, Deployment, Service, Ingress, ConfigMap, Secret, DaemonSet, StatefulSet, HPA (autoscaler), CRD (custom resource), Operator
- K8s ecosystem:
	- [[Helm]] (package manager)
	- [[Kustomize]] (overlay configs)
	- [[ArgoCD]]/[[Flux]] (GitOps)
	- cert-manager
	- [[Istio]], [[Linkerd]] (service mesh)
	- Cilium 
- Lighter than K8s:
	- [[HashiCorp Nomad]]
	- [[Amazon Elastic Container Service|AWS Elastic Container Service]]
	- Fly.io Machines
- [[Service Mesh]]: Sidecar proxies for inter-service traffic managmeent ([[Mutual TLS|mTLS]], retries, traces). [[Istio]], [[Linkerd]], [[HashiCorp Consul]]
- [[API Gateway]]s: [[Kong]], Tyk, [[Envoy]]-based (Emissary, Gloo), AWS API GAteway, GCP API Gatway, etc.


# [[Serverless]]/FaaS/Edge
- [[Amazon Lambda|AWS Lambda]]: Still the giant, now supports SnapSstart, container images, longer timouts
- [[Cloudflare Workers]]: V8 isolates, edge-first, very cheap, limited Node compatability
- [[Vercel Functions]] /Fluid compute. Node/Pytohn/Bun/Rust, instance reuse, fewer [[Cold Start]]s than classic Lambda.
Concepts to know:
- [[Cold Start]]
- [[Warm Pool]]
- Concurrency Models (per-instance vs reused)

# [[Infrastructure as Code]] (IaC)
- [[Terraform]] is still the default.
	- OpenTofu is the open-source fork after the license change.
- [[Pulumi]] is IAC in real languages like TS, Python, Go
- Tools like Ansible, Chef, Puppet around config management are declining.
- [[HashiCorp Packer]]: For image buildling

# Build/CI/CD
- [[Continuous Integration|CI]]: [[GitHub Actions]] is the default
- CD: [[ArgoCD]]/[[Flux]] ([[GitOps]] for K8s), Spinnaker, platform-built-ins (eg Vercel, Fly)
- Release Patterns: 
	- [[Blue/Green Deployment]]
	- [[Canary Deployment]]
	- [[Feature Flag]]s
		- [[LaunchDarkly]] leads for enterprise, Statsig, Posthog also seen
	- [[Trunk-Based Development]]
		- Version control management practice where developers merge small, frequent code updates into a central "trunk" (main/master branch), multiple times a day.
		- It avoids long-lived feature branches, reducing complex merge conflicts and enabling [[Continuous Integration]] (CI) and faster delivery. It requires robust automated testing and feature flags.

# Architectural Patterns
- [[Monolith]] vs [[Microservice]]s vs [[Modular Monolith]] ; then pendulum has swung back towards modular monolith as the default. Microservices reserved for org-scaling needs.
- [[Domain-Driven Design]] (DDD): [[Bounded Context]]s, [[Aggregate]]s, and [[Ubiquitous Language]]
- [[Hexagonal Architecture]]: Keeping core domain free of I/O
- [[Command Query Responsibility Segregation]] (CQRS): Separating the read and write models that are used.
- [[Event Sourcing]]: Store events, not state; often paired with [[Command Query Responsibility Segregation|CQRS]]
- [[Saga]] Pattern: Distributed transactions via compensating actions ([[Durable Execution Engine]]s like [[Temporal]] productize this, essentially).
- [[Outbox Pattern]]: Atomically write to DB + Queue
- [[Idempotency]]: Must-know, especially for HTTP APIs and message handlers.
- [[Backpressure]]: Slowing producers when consumers can't keep up.
- [[Bulkhead]]s and [[Circuit Breaker]]s (tools like Hystrix -> Resilience4j): Failure isolation.
- [[CAP Theorem]], [[PACELC]], [[Consistency]] models ([[Linearizability]], Sequential, Causal, Eventual)
- [[Two-Phase Commit]] (2PC): For [[Distributed Transaction]]s, usually avoided
- [[Distributed Consensus|Consensus]] algorithms; [[Raft]], [[Paxos]], [[Zab]]
- [[Conflict-Free Replicated Data Types]] (CRDTs) for collaborative/offline-first applications.

# Testing (backend-flavored)
- [[Unit Test]], [[Integration Test]], [[End to End Test]]
- [[Contract Test]]ing: [[Pact]]
- [[Property-Based Testing]]: [[Hypothesis]] (Python), etc. 
- [[Snapshot Testing]]: Fine for some things, dangerous when overused
- Load testing: k6, Gatling, Locust, vegeta, wrk
- Chaos engineering: Chaos Mesh, Litmus, ...
- Test Containers: [[Testcontainers]] (real DBs, queues in Docker for integration tests - Should be a default)
- Ephemeral environments: Neon branches, PlanetScale branches...

# Security
- [[OWASP Top 10]]: Know it
- Secrets management: [[HashiCorp Vault]], [[Amazon Secrets Manager]], Doppler
- SBOM: Software Bill of Materials
- Rate limiting: [[Token Bucket]], [[Leaky Bucket]],  Sliding window
	- Hosted: Upstash Ratelimit, Cloudflare


# AI/ML on the Backend
- LLM Gateways: [[OpenRouter]], Vercel AI Gateway, LiteLLM (self-host), Porkey, Helicone
- Vector search: [[pgvector]] first.
- Embedding pipelines: Chunking, indexing, [[Hybrid Search]] ([[BM25]] + [[Vector Search]] + [[Reranking|Reranker]])
	- Reranking: Cohere Rerank, Voyage AI
- [[Retrieval-Augmented Generation|RAG]] patterns and libraries: [[LlamaIndex]], [[LangChain]], ...
- Agent Runtimes: [[LangGraph]], ...
- [[Model Context Protocol|MCP Server]]s as backend integration layers
- LLM observability: [[Langfuse]], ... [[LangSmith]]
- Evals: First-class concern; treat eval suits like test suites


# Cloud Providers
- [[Amazon Web Services]]: Everything, complex, expensive
- [[Google Cloud Platform]]: Fewer services, cleaner DX
- Azure: Dogshit enterprise
- [[Cloudflare]]: Cheaper edge-first stack: [[Cloudflare Workers]], [[Cloudflare D1]] (SQLite), [[Cloudflare R2]] (S3-compatible with no egress fees), [[Cloudflare Durable Objects]], [[Cloudflare Queues]], [[Cloudflare Workers AI]]
- [[Fly.io]]: VMs near users, Postgres, simpler than AWS
- Render, [[Railway]], Northflank: [[Heroku]]-shaped [[Platform as a Service|PaaS]] replacements
- [[Vercel]]: Originally Next.js, but now a full compute platform.
- Hetzner: Cheap raw VMs, used for self-hosting


# Trends
- [[PostgreSQL|Postgres]] is the default, absorbing vector, queue, FTS, JSON, time-series, and your app data. Reach for a second DB ONLY when you have a real reason.
- [[Durable Execution Engine|Durable Execution]] is becoming a layer of the stack, like caches were 15 years ago.
- [[OpenTelemetry Protocol|OpenTelemetry]] won. If your instrumentation isn't OTel-emitting, it's a migration target.
- [[Modular Monolith]] is the new "boring default"; microservices are a scaling tool, not a starting architecture.
- [[Sidecar]]s are getting eaten by [[extended Berkeley Packet Filter|eBPF]] for service mesh and observability usecases.
- AI is now backend infrastructure: Vector store, gateawy, agent runtime, evals now belong in the backend architecture diagram.
- The Rust rewrite is real on the backend infra too (Linkerd-proxy, Redpanda, Polars, Pingora, sscache)
- Type-safety is leaking into runtime
- [[GitOps]] is the deployment default. Declarative state in git.
- Free-threaded Python (GILectomy) and JVM virtual threads make "I need to rewrite in Go for concurrency" a much weaker argument than five years ago.


# What to learn
- The senior interview lens is "Can this person reason about a system end-to-end and articulate tradeoffs". Calibrate to that!

1. Learn [[PostgreSQL|Postgres]] deeply ([[Multiversion Concurrency Control|MVCC]], indexes like [[B-Tree]], [[GIN Index]], [[BRIN Index]], query planning, connection pooling, replication, [[Isolation]] levels)
2. Caching patterns: [[Cache-Aside]], etc., [[Cache Stampede]] avoidance, [[Cache Invalidation Strategy|Invalidation]] strategies.
3. Queues and Durable Execution: [[Outbox Pattern|Outbox]], [[Idempotency]], [[Exactly Once]] effects, [[Saga]]s
4. Distributed systems primitives: [[CAP Theorem]], [[PACELC]], [[Distributed Consensus|Consensus]], [[Consistency]] models, [[Replication]] patterns, [[Partition|Partitioning]] ([[Consistent Hashing]], range, hash)
5. Observability: [[RED Method]], [[USE Method]], [[OpenTelemetry Protocol|OpenTelemetry]], what to alert on.
6. Auth: [[OAuth]] 2.1 + [[OpenID Connect|OIDC]] flows, Session vs Token, Authz models ([[Role-Based Access Control]] RBAC vs [[Relationship-Based Access Control]] ReBAC)
7. API Deisgn: [[Representational State Transfer|REST]] vs [[gRPC]] vs [[GraphQL]] tradeoffs, [[Idempotency]], Pagination, Versioning
8. Scaling Stories: [[Replication|Read Replica]]s, [[Sharding]], [[Write-Through Cache]], Event-Driven decoupling
	1. Know the order and smell that triggers each
9. A real LLM-augmented backend tech; [[Vector Search]], Gateways, Evals...




