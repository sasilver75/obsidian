

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
- Python: FatAPI (default for new APIs, async, Pydantic-typed, OpenAPI auto), Django+DRF, Flask (legacy)
- ...

# API Paradigms
- REST: Still the default
- GraphQL: Well past peak hype but stable in places where federation pays off
- [[gRPC]]: Standard for inter-service in polyglot environments. 
	- `Connect` is gaining ground.
- [[tRPC]]: Type-safe RPC for TS-only stakcs. No schema language, types flow through.
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
- [[Apache Cassandra|Cassandra]]: Wide-column, write-heavy, eventually cosnsitent. [[ScyllaDB]] is the C++ rewrite, much faster.
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
	- [[PostgreSQL|Postgres]] built-in FTS + [[paradeb]]/[[pg_search]]
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
