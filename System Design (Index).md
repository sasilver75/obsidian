


Terms
- [[Functional Requirement]]: A behavior or capability the system must provide to satisfy a user or business need. Use it to define what the system does before optimizing how well it does it.
- [[Non-Functional Requirement]]: A quality constraint on the system, such as latency, availability, durability, scalability, security, or cost. Treat these as design drivers because they usually determine architecture more than feature lists do.
- [[Service Level Objective]] (SLO): A target reliability or performance threshold that a service aims to meet over a defined window. Use it to turn vague reliability goals into measurable engineering and product tradeoffs.
- [[Service Level Indicator]] (SLI): A measured signal used to evaluate service health, such as request latency, error rate, or availability. Choose SLIs that reflect user-visible outcomes rather than only internal component health.
- [[Service Level Agreement]] (SLA): A contractual promise about service performance that usually includes consequences if targets are missed. Use SLAs sparingly and set them below internal SLOs so engineering has room to respond before customers are owed remedies.
- [[Latency]]: The time between a request or event starting and the corresponding response or result completing. Track high-percentile latency because averages often hide the slow paths users actually notice.
- [[Availability]]: The fraction of time a system can successfully serve valid requests. Improve it with redundancy, failover, and dependency isolation, but expect cost and complexity to rise quickly.
- [[Durability]]: The likelihood that committed data will survive failures without being lost. Prioritize it for user data, payments, audit logs, and other state that cannot be regenerated.
- [[Scalability]]: A system's ability to handle increased load by adding resources or improving architecture. Design for expected growth patterns rather than hypothetical infinite scale.
- [[Monolith]]: An application architecture where major business capabilities are built and deployed as one unit. Prefer it when the team or domain is small enough that simple deployment and local reasoning matter more than independent service scaling.
- [[Microservice|Service-Oriented Architecture]] / Microservices: An architecture that decomposes a system into separately deployed services with clear ownership and APIs. Use it when team boundaries, independent scaling, or fault isolation justify the operational overhead.
- [[Backend for Frontend]] (BFF): A backend layer tailored to the needs of a specific client experience such as web, mobile, or admin UI. Use it when clients have divergent data-shaping or workflow needs that would otherwise bloat shared APIs.
- [[API Gateway]]: A front door for APIs that centralizes routing, authentication, rate limiting, transformation, and policy enforcement. Use it to simplify client access, but avoid turning it into a large business-logic bottleneck.
- [[Load Balancing|Load Balancer]] ([[Transport Layer|Layer 4]], [[Application Layer|Layer 7]]): A component that distributes traffic across backend instances to improve availability, utilization, and fault tolerance. Use layer 4 for simple high-throughput routing and layer 7 when HTTP-aware routing or policy is needed.
- [[Reverse Proxy]]: A server that sits in front of backend services and forwards client requests while hiding internal service details. Use it for TLS termination, routing, caching, compression, or shielding origin services.
- [[Service Discovery]]: A mechanism that lets clients or routers find healthy service instances dynamically. Use it when instances change frequently because of autoscaling, deployments, or failures.
- [[Service Mesh]]: Infrastructure that manages service-to-service traffic, security, retries, observability, and policy outside application code. Use it when cross-service communication policy is complex enough to justify another operational layer.
- [[Rate Limiting|Rate Limiter]]: A control that restricts how often a client, user, or service can perform an action. Use it to protect shared resources and enforce fairness before overload reaches core dependencies.
- [[Representational State Transfer|REST]]: An architectural style for resource-oriented APIs using standard HTTP methods and representations. Use it for broadly understandable CRUD-like APIs where HTTP semantics and caching are useful.
- [[HTTP]]: The application-layer protocol used by the web for request-response communication between clients and servers. Understand it when designing APIs because caching, methods, status codes, and headers shape system behavior.
- [[Remote Procedure Call]]: A communication style where a client invokes a function-like operation on a remote service. Use RPC for action-oriented service calls, but remember network calls fail differently from local functions.
- [[gRPC]]: A high-performance RPC framework that typically uses Protocol Buffers and HTTP/2 for typed service communication. Use it for internal service APIs when strong contracts, streaming, and performance matter more than browser-native simplicity.
- [[Webhook]]: An HTTP callback sent by one system to notify another system that an event occurred. Use it for third-party integrations, but design for retries, duplicates, signature verification, and delayed delivery.
- [[WebSockets]]: A protocol that provides a persistent bidirectional connection between client and server. Use it for collaborative, realtime, or interactive applications where both sides need to send data at arbitrary times.
- [[Server-Sent Event]] (SSE): A browser-friendly HTTP mechanism for streaming one-way event updates from server to client. Use it when the server needs to push updates but the client does not need a full bidirectional channel.
- [[Idempotency]]: A property where repeating the same operation produces the same effect as doing it once. Use it for retries around payments, job submission, message handling, and any API where duplicate requests are plausible.
- [[Pagination]]: A technique for splitting large result sets into smaller chunks for retrieval and display. Use it to bound latency and payload size instead of letting clients request unbounded collections.
- [[Pagination|Cursor-Based Pagination]]: A pagination strategy that uses a stable cursor from the last seen item to fetch the next page, which works well for large or changing result sets. Prefer it for feeds and high-volume tables where inserts or deletes would make offsets unreliable.
- [[Pagination|Offset-Based Pagination]]: A pagination strategy that uses numeric offsets and limits, which is simple but can become slow or inconsistent as datasets grow or change. Use it for small, mostly static result sets or admin workflows where direct page numbers matter.
- Versioning: A compatibility strategy for evolving APIs, schemas, or services without breaking existing clients. Use it when consumers cannot all upgrade at once, and plan deprecation paths before old versions become permanent.
- [[Relational Database]]: A database that stores structured data in tables with rows, columns, constraints, and SQL queries. Prefer it when relationships, transactions, constraints, and ad hoc queries are central to the system.
- [[Key-Value Database|Key-Value Store]]: A database that stores values addressed by unique keys, optimized for simple lookups and writes. Use it for caches, sessions, profiles, or configuration where access is mostly by primary key.
- [[Document Database]]: A database that stores semi-structured records as documents, commonly JSON-like objects. Use it when records vary by shape and are usually read or written as whole aggregates.
- [[Wide-Column]]: A database model that stores sparse, distributed rows with flexible columns grouped into families. Use it for very large scale workloads with predictable query patterns and high write volume.
- [[Blob Storage|Object Storage]]: Storage for large immutable objects addressed by keys, commonly used for files, media, backups, and data lakes. Use it when data is accessed as whole objects rather than updated with fine-grained transactions.
- [[Index]]: A data structure that speeds reads by maintaining a searchable mapping from values to records. Add indexes around real query patterns, but budget for extra write cost and storage.
- [[Secondary Index]] ([[Local Secondary Index]], [[Global Secondary Index]]): An additional index over non-primary-key attributes that supports alternate query patterns. Use it when the primary key does not match important reads, but watch consistency and write amplification.
- [[Sharding]]: Splitting data across multiple nodes or partitions so load and storage can scale horizontally. Use it when a single node cannot handle the data or traffic, but plan carefully for hot keys and cross-shard operations.
- [[Replication]]: Copying data across multiple nodes to improve availability, durability, read capacity, or geographic locality. Use it to survive failures and scale reads, but reason explicitly about lag and consistency.
- [[Partition]]: A subset of data assigned to a storage unit, node, shard, or processing range. Choose partition keys that spread load evenly while preserving the queries the system must run efficiently.
- [[Replication|Read Replica]]: A replicated database copy used to serve reads without adding load to the primary writer. Use it for read scaling, but avoid it for reads that must immediately observe recent writes unless lag is handled.
- [[Write-Ahead Log]] (WAL): An append-only log written before data pages change so committed writes can be recovered after failures. It matters for durability, replication, point-in-time recovery, and understanding database write behavior.
- [[Cache]]: A faster storage layer that keeps frequently or recently used data close to where it is needed. Use it when repeated reads are expensive, but define freshness and invalidation rules before relying on it.
- [[Refresh-Ahead]]: Cache refreshes an entry in the background before it expires, usually when the key is hot or nearing its TTL. The goal is to avoid cache misses and keep frequently-read data fresh without making a user request wait for reload.
- [[Stale-While-Revalidate]]: The cache is allowed to return stale data shortly after expiry while it refreshes the entry in the background. "Don't make the user wait just because the cached value is technically expired."
- [[Cache Read Strategy|Cache Read Strategies]]: Ways for reading data from a cache (see below)
- [[Cache-Aside]]: A caching pattern where application code reads from cache first, loads from the source on miss, and writes the value back to cache. Use it when the application can tolerate cache misses and explicitly manage cache population.
- [[Read-Through Cache]]: A caching pattern where the cache itself loads missing data from the backing store. Use it when you want callers to treat the cache as the read interface, but ensure miss behavior and failures are visible.
- [[Cache Write Strategy]]: A policy that defines how writes update or bypass cache and backing storage. Pick the strategy based on the required balance among latency, durability, freshness, and operational simplicity.
- [[Write-Through Cache]]: A caching strategy where writes update the cache and backing store synchronously. Use it when read freshness matters and extra write latency is acceptable.
- [[Write-Back Cache|Write-Back]]: A caching strategy where writes update cache first and are flushed to backing storage later. Use it to reduce write latency, but only when data loss or delayed persistence is acceptable or mitigated.
- [[Write-Around Cache]]: A cache write strategy where writes bypass the cache and go directly to backing storage to avoid filling cache with cold data.
- [[Cache Invalidation Strategy|Cache Invalidation]]: The process of removing or refreshing cached data when it may no longer be correct. Treat it as a core design problem because stale cache behavior often defines correctness.
- [[Cache Eviction Strategy|Cache Eviction]]: The policy used to decide which cached items to remove when capacity is limited. Choose it based on access patterns, object size, and the cost of cache misses.
- [[Time to Live]] (TTL): A duration after which data, cache entries, tokens, or messages expire automatically. Use TTLs to bound staleness and cleanup work, but avoid relying on them for precise correctness.
- [[Cache Stampede]]/Thundering Herd: A failure mode where many clients regenerate the same expired or missing cache entry at once. Prevent it with request coalescing, jittered TTLs, locks, or stale-while-revalidate behavior.
- [[Content Delivery Network]] (CDN): A geographically distributed caching network that serves static or cacheable content close to users. Use it to reduce latency and origin load, but design cache keys, purging, and authorization carefully.
- [[Message Queue]]: A system that buffers messages between producers and consumers to decouple work and absorb bursts. Use it for asynchronous work, smoothing spikes, and retryable processing outside the request path.
- [[Publish-Subscribe|Pub Sub]]: A messaging pattern where publishers send events to topics and subscribers receive matching events. Use it when multiple downstream consumers need to react independently to the same event.
- [[Kafka]]: A distributed event streaming platform built around durable, partitioned, ordered logs. Use it when consumers need replayable streams, ordering within partitions, and high-throughput event pipelines.
- [[Amazon SNS|AWS SNS]], [[Amazon SQS|AWS SQS]]: AWS SNS provides pub-sub fanout while AWS SQS provides durable queue-based message buffering. Use SNS for broadcast-style notification and SQS for pull-based work queues.
- [[Dead Letter Queue]]: A queue that stores messages that could not be processed successfully after retries. Use it to keep bad messages from blocking progress while preserving them for inspection and repair.
- [[Retry]]: Repeating a failed operation in hopes that a transient error has cleared. Use it only for failures that are likely transient, and pair it with idempotency and backoff.
- [[Backoff]]: A retry strategy that increases delay between attempts to reduce load and improve recovery odds. Use jitter with backoff so many clients do not retry in synchronized waves.
- [[Transactional Outbox Pattern]]: A pattern that records outbound events in the same transaction as state changes so they can be published reliably later. Use it when a service must update its database and emit an event without losing one side of the change.
- [[Change Data Capture]] (CDC): Capturing database changes from logs or tables so downstream systems can react or replicate data. Use it when downstream systems need reliable updates without every writer manually publishing events.
- [[Consistency]]: The degree to which reads observe the latest or expected state across replicas, transactions, or distributed components. Define the exact consistency needed per workflow instead of treating stronger consistency as universally better.
- [[Strong Consistency]]: A guarantee that reads reflect the latest successful write according to the system's ordering rules. Use it when stale reads would violate correctness, money movement, permissions, or user trust.
- [[Eventual Consistency]]: A guarantee that replicas converge if no new updates occur, while reads may temporarily see stale data. Use it when availability, latency, or scale matter more than immediately perfect reads.
- [[CAP Theorem]]: The principle that a distributed system under network partition must choose between consistency and availability. Use it to reason about partition behavior, not as a blanket excuse for vague tradeoffs.
- [[Distributed Consensus|Consensus]]: A protocol by which distributed nodes agree on a value, order, or leadership despite failures. Use it for replicated metadata, leader election, and critical coordination where split decisions are dangerous.
- [[Leader Election]]: A process for choosing one node to coordinate work, accept writes, or manage cluster decisions. Use it when exactly one active coordinator is needed, but plan for failover and fencing.
- [[Distributed Lock]]: A coordination primitive that attempts to ensure only one distributed worker holds a critical section at a time. Use it cautiously because clock skew, pauses, and network partitions can make lock ownership subtle.
- [[Two-Phase Commit]] (2PC): A distributed transaction protocol with a prepare phase and a commit phase coordinated across participants. Use it only when atomic cross-resource commit is required and blocking during coordinator failure is acceptable.
- [[Three-Phase Commit]] (3PC): A distributed commit protocol that adds an extra phase to reduce some blocking behavior of 2PC. It is mostly useful as a conceptual contrast because real systems often prefer consensus or sagas.
- [[Saga]]: A long-running workflow composed of local transactions with compensating actions for rollback-like recovery. Use it when a business process spans services and can tolerate explicitly modeled compensation.
- [[Circuit Breaker]]: A resilience pattern that stops calling a failing dependency temporarily to prevent cascading failures. Use it around unreliable or overloaded dependencies so callers fail fast and give the dependency time to recover.
- [[Backpressure]]: A mechanism for slowing producers when consumers or downstream systems cannot keep up. Use it in queues, streams, APIs, and pipelines to prevent unbounded memory growth and cascading overload.
- [[Observability]]: The ability to understand a system's internal state from external signals such as logs, metrics, and traces. Design it early because production debugging depends on signals captured before the incident happens.
- [[Logging]]: Recording discrete events and contextual details to support debugging, auditing, and operations. Use logs for high-cardinality context and narratives, but avoid leaking secrets or relying on logs for every metric.
- [[Metrics]]: Numeric time-series measurements that summarize system behavior over time. Use metrics for alerting, dashboards, capacity planning, and trend detection.
- [[Tracing]]: Tracking a request or workflow across services to understand latency, dependencies, and failure paths. Use it in distributed systems where a single user action crosses many services or async boundaries.
- [[OpenTelemetry Protocol|OpenTelemetry]]: A standard ecosystem for collecting, processing, and exporting telemetry such as traces, metrics, and logs. Use it to avoid vendor lock-in and to standardize instrumentation across services.
- [[Horizontal Scaling]]: Increasing capacity by adding more instances, nodes, or workers. Prefer it when the workload can be partitioned or replicated cleanly across machines.
- [[Vertical Scaling]]: Increasing capacity by giving an existing node more CPU, memory, disk, or network resources. Use it for simple scaling or bottleneck relief, but expect hard ceilings and larger failure blast radius.
- [[Authentication]]: Verifying the identity of a user, service, or device. Use it before authorization decisions, but do not treat identity alone as permission.
- [[Authorization]]: Deciding what an authenticated or anonymous actor is allowed to access or do. Model it close to the protected resource because permissions often depend on object ownership, tenant, role, or context.
- [[OAuth]] (OAuth 2.0): A delegated authorization framework that lets clients access resources with scoped tokens. Use it for delegated API access, but pair it with OIDC when the goal is user login.
- [[OpenID Connect]] (OIDC): An identity layer on top of OAuth 2.0 that standardizes login and user identity claims. Use it for SSO and federated identity, but validate issuer, audience, nonce, and signatures.
- [[JSON Web Token]] (JWT): A compact signed token format for carrying claims between systems. Use it when stateless verification is useful, but keep lifetimes short because revocation is hard.
- [[JSON Web Key Set]] (JWKS): A JSON document that publishes public keys used to verify signed tokens. Use it for key discovery and rotation, but cache it carefully so old and new keys overlap safely.
- [[Platform-Agnostic Security Token]] (PASETO): A secure token format designed as a safer alternative to common JWT pitfalls. Use it when you control both issuer and verifier and do not need JWT ecosystem compatibility.
- [[Mutual TLS]] (mTLS): A TLS mode where both client and server authenticate each other with certificates. Use it for service-to-service identity or high-trust clients, but plan for certificate issuance, rotation, and debugging.
- [[Service-to-Service Authentication]] (S2S Auth): Authentication that proves one backend service is allowed to call another. Use it to reduce implicit trust inside a network boundary.
- [[Machine-to-Machine Authentication]] (M2M Auth): Authentication for non-human clients such as services, jobs, agents, and devices. Use it for automation and backend workloads, but scope credentials narrowly because there is no human approval step at runtime.
- [[Durable Execution Engine]]: A workflow runtime that persists progress so long-running tasks can survive crashes and retries. Use it when workflows span time, services, retries, timers, or human approval.
- [[ACID]]: A set of transaction properties: atomicity, consistency, isolation, and durability. Use it to reason about database correctness before splitting state across services or asynchronous workflows.
- [[Transaction]]: A group of operations treated as one logical unit of work. Use transactions when partial completion would leave data in an invalid or misleading state.
- [[Distributed Transaction]]: A transaction that coordinates changes across multiple databases, services, partitions, or resource managers. Avoid it when sagas or idempotent async workflows can express the business process with less coupling.
- [[Multiversion Concurrency Control]] (MVCC): A concurrency technique that keeps multiple record versions so readers and writers can proceed with less blocking. It is relevant when diagnosing isolation behavior, vacuum pressure, stale snapshots, or read/write contention.
- [[Optimistic Concurrency Control]] (OCC): A concurrency strategy that allows work to proceed and checks for conflicts before commit. Use it when conflicts are rare and retrying failed writes is acceptable.
- [[Pessimistic Concurrency Control]] (PCC): A concurrency strategy that prevents conflicts by locking data before operations proceed. Use it when conflicts are common or retries would be expensive or dangerous.
- [[Two-Phase Locking]] (2PL): A locking protocol where transactions acquire locks before releasing any, then release them without acquiring new ones. It matters when strict serializability is needed but lock contention and deadlocks must be managed.
- [[Strong Consistency|Linearizability]]: A consistency guarantee where each operation appears to occur atomically at one instant between invocation and response. Use it for single-object operations where clients need an immediately current result.
- [[Serializable Isolation]]: An isolation guarantee where concurrent transactions produce the same result as some serial order. Use it when transaction correctness matters more than maximum concurrency.
- [[Read-your-Writes Consistency]]: A guarantee that a client can read its own previously successful writes. Use it for user-facing workflows where saving something and not seeing it immediately would feel broken.
- [[Causal Consistency]]: A guarantee that causally related operations are observed in causal order. Use it in collaborative or replicated systems where users should not see effects before their causes.
- [[Consistent Hashing]]: A partitioning technique that maps keys and nodes onto a ring to reduce remapping when nodes change. Use it for caches and distributed stores where nodes are added or removed often.
- [[Hot Spot]]/[[Hot Spot|Hot Key]]: A key, partition, node, or resource that receives disproportionate load and becomes a bottleneck. Look for it when average load looks safe but one shard, user, tenant, or item is overloaded.
- [[Full-Text Search Index]]: An index optimized for searching natural-language text by terms, tokens, ranking, and relevance. Use it when users search text semantically or fuzzily rather than filtering exact structured fields.
- [[Geospatial Index]]: An index optimized for location-based queries such as containment, intersection, nearest-neighbor, or distance search. Use it for maps, proximity search, delivery zones, and location-aware matching.
- [[B-Tree]]: A balanced tree index optimized for ordered lookups, range scans, and sorted access. Prefer it for equality plus range queries over ordered scalar fields.
- [[Hash Index]]: An index optimized for equality lookups by hashing indexed values. Use it when exact match dominates and range ordering is not needed.
- [[GIN Index]]: A PostgreSQL inverted index useful for multi-valued fields such as arrays, JSONB, and full-text search. Use it when a row may contain many searchable terms or keys.
- [[BRIN Index]]: A PostgreSQL block-range index that summarizes value ranges across physical table blocks. Use it for very large tables where values correlate with physical order, such as timestamps in append-only data.
- [[Materialized View]]: A stored query result that can speed reads at the cost of refresh complexity and staleness. Use it when expensive derived data is read often and can tolerate controlled lag.
- [[Connection Pool]]: A managed set of reusable database or network connections that reduces connection setup overhead. Use it to protect databases from connection storms and to amortize handshake cost.
- [[Sticky Session]]: A load-balancing behavior that routes a client to the same backend instance across requests. Use it only when local instance state is unavoidable, because it weakens load distribution and failover.
- [[Serverless]]: A deployment model where infrastructure scaling and server management are abstracted behind event-driven or managed compute. Use it for spiky or event-driven workloads, but check cold starts, time limits, and provider coupling.
- [[Role-Based Access Control]] (RBAC): An authorization model that grants permissions through roles assigned to users or principals. Use it when permissions align with stable job functions or administrative roles.
- [[Attribute-Based Access Control]] (ABAC): An authorization model that evaluates attributes of subjects, resources, actions, and context. Use it when access depends on dynamic properties such as department, tenant, region, device, or data classification.
- [[Relationship-Based Access Control]] (ReBAC): An authorization model that grants access based on relationships between users and resources. Use it for collaboration graphs, sharing models, organizations, and social-style permissions.
- [[Row-Level Security]] (RLS): A database feature that restricts which rows a user or role can read or modify. Use it to enforce tenant or user boundaries close to the data, but test policies carefully because mistakes can hide or expose rows.
- [[Cross-Site Request Forgery]] (CSRF): An attack where a browser is tricked into making an authenticated request to another site. It matters for cookie-authenticated web apps, especially state-changing requests.
- [[Cross-Origin Resource Sharing]] (CORS): A browser security mechanism that controls which origins may access cross-origin HTTP responses. Configure it when browser clients call APIs across origins, but do not treat it as server-side authorization.
- [[Tail Latency]]: High-percentile latency, such as p95 or p99, that captures the slowest user-visible requests. Optimize it when user experience or upstream timeouts are dominated by outliers.
- [[Bulkhead]]: A resilience pattern that isolates resources so one failing component cannot exhaust the entire system. Use it to contain blast radius across tenants, dependencies, pools, queues, or service groups.
- [[Graceful Degradation]]: Designing a system to keep core functionality working when dependencies or capacity are impaired. Use it when partial results or reduced features are better than total failure.
- [[Health Check]]: A probe that reports whether a service or instance is healthy enough for operational use. Use it for monitoring and automation, but make sure it reflects meaningful dependency and runtime state.
- [[Readiness Check]]: A probe that reports whether an instance is ready to receive traffic. Use it during startup, deployment, and dependency outages so traffic is not routed to unready instances.
- [[Liveness Check]]: A probe that reports whether an instance is alive or should be restarted. Keep it narrow so slow dependencies do not cause unnecessary restart loops.
- [[Timeout]]: A maximum waiting period after which an operation is abandoned or treated as failed. Set timeouts on every network dependency so callers do not wait indefinitely and exhaust resources.
- [[Session|User Session]]: Server or client state representing a user's authenticated interaction over time. Use sessions when user continuity is needed, but design expiration, revocation, and storage deliberately.
- [[User Access Token]]: A credential used by a client to access resources on behalf of a user. Keep it short-lived and scoped because a leaked bearer token can usually be used immediately.
- [[Refresh Token]]: A long-lived credential used to obtain new access tokens without asking the user to log in again. Store and rotate it more carefully than access tokens because compromise lasts longer.
- [[At Least Once]] Delivery: A delivery guarantee where a message is delivered one or more times and consumers must handle duplicates. Use it when losing work is worse than processing duplicates.
- [[At Most Once]] Delivery: A delivery guarantee where a message is delivered zero or one time and may be lost. Use it only when low latency or simplicity matters more than guaranteed processing.
- [[Exactly Once]] Delivery: A processing goal where each message's effect is applied once despite retries or failures. Treat it as an end-to-end design property, usually requiring idempotent writes and careful state management.
- [[Blue-Green Deployment]]: A release strategy that switches traffic between two complete production environments. Use it when fast rollback and environment-level separation justify duplicate capacity.
- [[Canary Deployment]]: A release strategy that gradually exposes a new version to a small and then growing portion of traffic. Use it when production signals can safely validate a change before full rollout.
- [[Feature Flag]]: A runtime switch that enables, disables, or varies behavior without redeploying code. Use it for gradual rollout, kill switches, experiments, and separating deploy from release.
- [[Active-Active]]: A high-availability topology where multiple sites or instances serve production traffic simultaneously. Use it for high availability and low regional latency, but design for conflict handling and traffic steering.
- [[Active-Passive]]: A high-availability topology where a standby environment takes over when the primary fails. Use it when simpler failover is acceptable and the standby lag or activation time meets recovery goals.
- [[Domain Name Service]] (DNS): A distributed naming system that maps domain names to IP addresses, service records, and other routing metadata.
- [[HTTPS]]: HTTP over TLS, providing encrypted transport, integrity protection, and server authentication for web traffic.
- [[HTTP 1.1]]: A text-based HTTP version with persistent connections and request pipelining limits that can cause connection-level blocking.
- [[HTTP 2]]: A binary multiplexed HTTP version that improves connection reuse and request concurrency while still depending on TCP ordering.
- [[HTTP 3]]: An HTTP version built on QUIC over UDP to reduce connection setup cost, improve migration, and avoid TCP head-of-line blocking.
- [[Head-of-Line Blocking]]: A performance failure mode where one blocked request, packet, or queue item prevents later independent work from progressing.
- [[Transport Layer Security]] (TLS): A cryptographic protocol that protects network connections with encryption, integrity checks, and endpoint authentication.
- [[TLS Termination]]: Decrypting TLS at a proxy, load balancer, or edge service before forwarding traffic to backend systems.
- [[TCP Termination]]: Ending the client TCP connection at an intermediary so backend systems use separate server-side connections.
- [[Proxy]]: An intermediary that forwards requests and responses between clients and servers while applying routing, policy, caching, or inspection.
- [[Forward Proxy]]: A proxy used by clients to reach external services, often for egress control, anonymity, filtering, or policy enforcement.
- [[Origin Server]]: The authoritative backend server that holds or generates content fetched by caches, CDNs, and edge services.
- [[Edge Server]]: A server close to users that handles caching, routing, compute, or security checks to reduce latency and origin load.
- [[GraphQL]]: An API query language and runtime where clients request typed fields from a schema instead of calling fixed resource endpoints.
- [[PostgreSQL]]: A relational database known for SQL support, transactions, indexing, JSON features, replication, and extensibility.
- [[Redis]]: An in-memory data store commonly used for low-latency caches, counters, queues, locks, and transient state.
- [[Online Transactional Processing]] (OLTP): A workload pattern optimized for many small, low-latency transactions that update current operational state.
- [[Online Analytical Processing]] (OLAP): A workload pattern optimized for large analytical queries, aggregations, and historical reporting.
- [[Data Warehouse]]: A centralized store optimized for analytical querying and reporting over structured business data.
- [[Data Lakehouse]]: A data architecture that combines low-cost data lake storage with warehouse-style tables, transactions, and query performance.
- [[Time-Series Database]]: A database optimized for timestamped measurements, retention windows, downsampling, and time-range queries.
- [[Distributed File System]]: A storage system that spreads file data across multiple machines while presenting a shared namespace or API.
- [[Physical Replication]]: Replication that copies low-level storage or WAL changes so a replica mirrors the primary database's physical state.
- [[Logical Replication]]: Replication that publishes table-level or row-level changes in database terms for selective replication and subscription.
- [[Composite Index]]: An index over multiple columns or fields that supports queries matching the index's combined key order.
- [[Bloom Filter]]: A probabilistic set-membership structure that can return false positives but not false negatives.
- [[Count-Min Sketch]]: A probabilistic frequency structure that estimates item counts with small memory and bounded overestimation.
- [[HyperLogLog]]: A probabilistic cardinality structure that estimates distinct counts with small, fixed memory.
- [[Inverted File Index]]: An index that maps terms or values to the documents or records that contain them.
- [[Event]]: An immutable record that something meaningful happened in the system or business domain.
- [[Stream Processing]]: Continuous processing of ordered or semi-ordered event streams as data arrives.
- [[Complex Event Processing]]: Detecting higher-level patterns by correlating multiple events across time windows, rules, or sequences.
- [[Kafka Connect]]: A Kafka integration framework for moving data between Kafka topics and external systems using reusable connectors.
- [[Command Query Responsibility Segregation]] (CQRS): An architecture pattern that separates write models from read models so each can be optimized independently.
- [[Schema Registry]]: A service that stores and validates versions of schemas used by events, messages, or APIs.
- [[Event Sourcing]]: A persistence pattern where the source of truth is an append-only event log and current state is derived from those events.
- [[Cloud Computing]]: A model for using compute, storage, networking, and managed services as on-demand resources rather than self-owned infrastructure.
- [[Infrastructure as a Service]] (IaaS): A cloud model that provides virtualized compute, storage, and networking primitives for users to manage.
- [[Platform as a Service]] (PaaS): A cloud model that provides managed application platforms so teams deploy code without managing most infrastructure.
- [[Software as a Service]] (SaaS): A delivery model where users consume complete software applications hosted and operated by a provider.
- [[Backend-as-a-Service]] (BaaS): A platform model that provides managed backend capabilities such as auth, databases, storage, and functions.
- [[Cloudflare Workers]]: An edge compute platform for running serverless JavaScript or WebAssembly close to users on Cloudflare's network.
- [[Cloudflare Durable Objects]]: A Cloudflare primitive that provides single-threaded, stateful coordination for a named object or key.
- [[Cloudflare D1]]: A managed serverless SQL database from Cloudflare based on SQLite.
- [[Cloudflare R2]]: An S3-compatible object storage service from Cloudflare designed for low-cost storage and reduced egress friction.
- [[Google Cloud Run]]: A managed serverless container platform that runs stateless containers behind HTTP or event-driven triggers.
- [[Amazon CloudFront]]: AWS's content delivery network for caching and delivering content from edge locations.
- [[Amazon Application Load Balancer]]: An AWS layer 7 load balancer for HTTP and HTTPS routing based on hosts, paths, headers, and targets.
- [[Amazon Network Load Balancer]]: An AWS layer 4 load balancer for high-throughput TCP, UDP, and TLS traffic.
- [[Amazon EventBridge]]: An AWS event bus service for routing events between applications, SaaS tools, and AWS services.
- [[Virtual Private Cloud]] (VPC): An isolated virtual network in a cloud provider where subnets, routing, and security boundaries are configured.
- [[Virtual Private Cloud Peering]]: A private network connection between VPCs that lets resources communicate without traversing the public internet.
- [[Internet Gateway]]: A cloud networking component that allows resources in a public subnet to communicate with the internet.
- [[Long Polling]]: A client-server pattern where the server holds a request open until data is available or a timeout occurs.
- [[Short Polling]]: A client-server pattern where the client repeatedly asks for updates at fixed intervals.
- [[Web Authentication API]] (WebAuthn): A browser standard for phishing-resistant authentication using public-key credentials such as passkeys or hardware keys.
- [[Certificate Authority]]: A trusted entity that issues and signs certificates binding identities to public keys.
- [[Certificate Chain]] / Chain of Trust: A sequence of certificates linking a leaf certificate to a trusted root certificate authority.
- [[Security Assertion Markdown Language|Security Assertion Markup Language]] (SAML): An XML-based federation protocol for exchanging authentication and authorization assertions between identity providers and service providers.
- [[Auth0]]: An identity platform that provides hosted login, token issuance, federation, and user management features.
- [[Service Token]]: A non-user credential used by services, jobs, or automation to authenticate to APIs.
- [[Application Performance Monitoring]] (APM): Tooling for measuring application performance, errors, dependencies, and production behavior.
- [[Golden Signals of Monitoring]]: The four core service health signals of latency, traffic, errors, and saturation.
- [[RED Method]]: An observability method that monitors request rate, errors, and duration for request-driven services.
- [[USE Method]]: An observability method that monitors utilization, saturation, and errors for each resource.
- [[N + 1 Query Problem]]: A performance problem where fetching a collection triggers one query for the collection and one additional query per item.
- [[Request Coalescing]]: Combining concurrent identical requests so only one expensive backend call or cache fill is performed.
- [[Cache Warming]]: Proactively loading cache entries before user traffic needs them.
- [[Distributed Cache]]: A shared cache accessed by multiple application instances or services.
- [[In-Process Cache]]: A cache stored inside a single application process for very low latency at the cost of locality and duplication.
- [[Client-Side Cache]]: A cache stored in a browser, mobile app, SDK, or other consuming client to reduce network calls.
- [[Fencing Token]]: A monotonically increasing token used with distributed locks so stale lock holders can be rejected.
- [[Consumer Group]]: A set of consumers that share work from a stream or topic so partitions or messages are distributed among members.
- [[Stream Offset]]: A position in an ordered stream that tracks where a consumer has read or committed progress.
- [[Long-Running Task]]: An operation that outlasts a normal request-response cycle and needs asynchronous tracking, retries, or cancellation.
- [[Batching]]: Grouping operations together to amortize overhead and improve throughput at the cost of latency or complexity.
- [[Load Shedding]]: Intentionally rejecting or degrading lower-priority work to keep core system capacity available under overload.
- [[Autoscaling]]: Automatically adjusting compute or service capacity based on demand, schedules, or resource signals.
- [[Admission Control]]: A mechanism that decides whether to accept new work based on current capacity, priority, or policy.
- [[Brownout]]: Temporarily disabling nonessential functionality during overload so critical paths remain available.
- [[Priority Queueing]]: Scheduling higher-priority work ahead of lower-priority work when capacity is constrained.
- [[Graceful Overload]]: A design approach where a system degrades or rejects selectively instead of collapsing under excess load.
- [[Retry Storm]]: An overload failure mode where many clients retry at once and amplify pressure on a struggling dependency.
- [[Cascading Failure]]: A failure pattern where one component's failure triggers failures in dependent or neighboring components.
- [[Capacity Planning]]: Estimating the resources needed to meet workload, latency, and reliability targets under normal, peak, and failure conditions.
- [[Queries Per Second]] (QPS): A rate measure for how many queries or requests a system handles per second.
- [[Throughput]]: The amount of work or data a system completes per unit of time.
- [[Latency Budget]]: An allocation of acceptable latency across client work, network hops, services, dependencies, and storage.
- [[Burst Traffic]]: A short-lived spike in demand above normal baseline traffic.
- [[Peak-to-Average Ratio]]: The ratio between peak load and average load that shapes provisioning, autoscaling, and buffering needs.
- [[Multi-Tenancy]]: An architecture where one system serves multiple tenants while isolating their data, configuration, and resource usage.
- [[Tenant Isolation]]: Mechanisms that prevent one tenant from accessing, corrupting, or degrading another tenant's data or resources.
- [[Noisy Neighbor Problem]]: A multi-tenant failure mode where one tenant's workload harms performance or availability for others.
- [[Control Plane]]: The part of a system that configures, schedules, manages, or orchestrates resources rather than handling primary user traffic.
- [[Data Plane]]: The part of a system that handles actual user, request, packet, or data traffic according to control-plane configuration.
- [[Disaster Recovery]]: The planning and mechanisms used to restore service and data after a major failure, region loss, or destructive mistake.
- [[Recovery Point Objective]] (RPO): The maximum acceptable amount of data loss measured as time before a disruption.
- [[Recovery Time Objective]] (RTO): The maximum acceptable time to restore service after a disruption.
- [[GeoDNS]]: DNS-based routing that uses requester location or network metadata to return a regionally appropriate endpoint.
- [[GeoRouting]]: Traffic routing that selects an endpoint based on geography, latency, compliance, or availability.
- [[Anycast]]: A routing technique where the same IP prefix is advertised from multiple locations so networks route users to a nearby site.
- [[Split Brain]]: A distributed failure mode where partitioned nodes or regions each believe they are authoritative.
- [[Conflict Resolution]]: Rules or mechanisms that reconcile concurrent, divergent, or replicated updates into an accepted state.
- [[Clock Skew]]: Differences between machine clocks that can break ordering, expiry, lease, and consistency assumptions.
- [[Quorum]]: The minimum number of nodes, replicas, or votes required to accept a read, write, or decision.
- [[Quorum Read]]: A read that requires enough replica responses to satisfy a quorum rule.
- [[Quorum Write]]: A write that requires enough replica acknowledgements to satisfy a quorum rule.
- [[Schema Evolution]]: Changing data, message, or API schemas over time while preserving compatibility for existing readers and writers.
- [[Online Schema Migration]]: A database schema change performed while the application remains available.
- [[Backfill]]: A batch process that fills, repairs, or recomputes historical data after a schema, model, or pipeline change.
- [[Dual Write Problem]]: Writing the same logical change to two systems, often creating consistency and rollback risks.
- [[Tombstone]]: A marker indicating that a record or key was deleted without immediately removing every trace of it.
- [[Compaction]]: Rewriting logs, files, or storage segments to remove obsolete versions, deleted records, or redundant data.
- [[Data Retention]]: A policy that defines how long data is stored before archival or deletion.
- [[Right to Be Forgotten]]: A privacy requirement to delete a person's data on request where legally or contractually required.
- [[Runbook]]: A documented operational procedure for diagnosing, responding to, or recovering from a known situation.
- [[Incident Management]]: The process for detecting, coordinating, mitigating, and communicating production incidents.
- [[Postmortem]]: A retrospective analysis of an incident that records impact, causes, lessons, and follow-up actions.
- [[Change Management]]: The process for planning, reviewing, approving, and deploying changes with controlled operational risk.
- [[Rollback Strategy]]: A plan for returning a system to a prior stable version or state after a bad change.
- [[Threat Model]]: A structured analysis of assets, attackers, entry points, risks, and mitigations for a system.
- [[Secrets Management]]: The secure storage, access, rotation, and auditing of credentials such as passwords, tokens, and private keys.
- [[Key Rotation]]: Replacing or revoking cryptographic keys and credentials on a planned or emergency basis.
- [[Principle of Least Privilege]]: Granting each user, service, or process only the minimum permissions needed to perform its job.
- [[Encryption at Rest]]: Encrypting stored data so stolen disks, backups, snapshots, or storage objects do not reveal plaintext.
- [[Encryption in Transit]]: Encrypting data while it crosses networks so intermediaries cannot read or tamper with it.



_______________

# Common Patterns


| Pattern                        | What it solves                                | How it’s accomplished                                                                                    |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Reservation / hold with TTL    | Temporarily claims scarce inventory.          | Create a pending reservation row with an expiry time, then confirm or release it via background cleanup. |
| Idempotency key                | Prevents duplicate side effects from retries. | Store request keys with their final result, and return the stored result on duplicate keys.              |
| Unique ID generation           | Creates identifiers without collisions.       | Use UUID/ULID, DB sequences, or Snowflake-style IDs combining timestamp, machine ID, and counter.        |
| Optimistic concurrency         | Handles conflicting writes without locks.     | Update only when a version/timestamp matches, then retry or reject on mismatch.                          |
| Pessimistic locking / leases   | Prevents concurrent mutation.                 | Acquire a DB lock, Redis lock, or lease with expiry before changing shared state.                        |
| Rate limiting                  | Controls excessive traffic.                   | Track request counts or tokens per user/IP/key in Redis or local counters.                               |
| Fan-out on write               | Makes reads fast for feeds/notifications.     | When an event occurs, write copies into each recipient’s inbox/feed.                                     |
| Fan-out on read                | Avoids huge write amplification.              | Store original events and assemble the recipient’s view at request time.                                 |
| Queue-based async work         | Moves slow work out of request paths.         | Enqueue jobs in Kafka/SQS/RabbitMQ/etc. and process them with workers.                                   |
| Pub/sub events                 | Lets many systems react independently.        | Publish events to a topic; subscribers consume and handle their own side effects.                        |
| Compare-and-swap               | Makes conditional updates atomic.             | Use atomic DB updates, Redis commands, or versioned writes like `WHERE version = ?`.                     |
| Write-ahead log                | Preserves intent before mutation.             | Append the operation to durable storage before applying it to state.                                     |
| Outbox pattern                 | Reliably publishes events from DB changes.    | Write the business row and outbox row in one transaction, then a relay publishes the event.              |
| Saga pattern                   | Coordinates multi-step workflows.             | Chain local transactions and define compensating actions for rollback-like recovery.                     |
| Two-phase commit               | Gives cross-resource atomicity.               | A coordinator asks participants to prepare, then tells all to commit or abort.                           |
| Event sourcing                 | Stores history as source of truth.            | Append immutable domain events and rebuild current state by replaying them.                              |
| CQRS                           | Separates write logic from read shape.        | Use one model for commands and separate projections/tables/indexes for queries.                          |
| Materialized views             | Speeds up expensive reads.                    | Precompute query results and refresh them on schedule or from change events.                             |
| Deduplication table            | Prevents duplicate message processing.        | Record processed message IDs and skip work when an ID already exists.                                    |
| Soft deletes                   | Keeps recoverable deleted data.               | Mark rows with `deleted_at` or status instead of physically deleting them.                               |
| Caching                        | Reduces latency and backend load.             | Store frequently used data in Redis, CDN, memory, or edge caches.                                        |
| Cache-aside                    | Simple app-managed caching.                   | Read cache first, fetch from DB on miss, then write the value into cache.                                |
| Write-through cache            | Keeps cache updated on writes.                | Write to cache and backing store together through a shared code path.                                    |
| Read replicas                  | Scales database reads.                        | Replicate primary DB data to secondary nodes and route read queries there.                               |
| Search index                   | Supports text/flexible querying.              | Copy searchable documents into Elasticsearch/OpenSearch/etc. and query the index.                        |
| Denormalization                | Avoids expensive joins or remote reads.       | Duplicate selected fields into read tables, documents, or service-local copies.                          |
| Pagination                     | Handles large result sets.                    | Use cursors or stable sort keys instead of returning everything at once.                                 |
| Precomputation                 | Trades write/storage for fast reads.          | Compute rankings, aggregates, feeds, or recommendations ahead of request time.                           |
| CDN edge caching               | Serves content near users.                    | Cache static or cacheable dynamic responses at CDN points of presence.                                   |
| Sharding / partitioning        | Splits data and traffic.                      | Route records to partitions by hash, tenant, region, user ID, or time.                                   |
| Hot key mitigation             | Prevents one key from overloading a shard.    | Split hot keys, salt counters, cache aggressively, or special-case large actors.                         |
| Batching                       | Improves throughput.                          | Combine many operations into one DB write, API call, queue publish, or network request.                  |
| Backpressure                   | Protects overloaded systems.                  | Bound queues and reject, delay, or slow producers when consumers fall behind.                            |
| Write coalescing               | Reduces redundant writes.                     | Merge repeated updates and persist only the latest value or aggregate delta.                             |
| Log-structured ingestion       | Handles high write volume.                    | Append events quickly to a log, then process/compact/index them later.                                   |
| Time-based partitioning        | Manages logs/events efficiently.              | Partition tables or objects by hour/day/month and expire old partitions.                                 |
| Retries with backoff           | Survives transient failures.                  | Retry after increasing delays, usually with jitter and a max attempt limit.                              |
| Circuit breaker                | Avoids hammering failing dependencies.        | Track failures and temporarily fail fast once a threshold is crossed.                                    |
| Timeouts                       | Bounds waiting.                               | Set explicit deadlines for network calls, DB queries, locks, and jobs.                                   |
| Bulkheads                      | Isolates failures.                            | Give subsystems separate thread pools, queues, connection pools, or capacity limits.                     |
| Graceful degradation           | Keeps partial service alive.                  | Return cached, simplified, or partial responses when dependencies are unavailable.                       |
| Health checks                  | Detects bad instances.                        | Expose readiness/liveness endpoints and remove failing instances from rotation.                          |
| Leader election                | Picks one coordinator.                        | Use consensus systems, leases, or lock services like ZooKeeper/etcd/Consul/Redis.                        |
| Failover                       | Moves work away from failure.                 | Promote replicas, reroute traffic, or switch regions using load balancers/DNS/control planes.            |
| Dead-letter queue              | Captures repeatedly failed work.              | Move messages to a separate queue after max retries for inspection or replay.                            |
| Replay                         | Rebuilds state after failure.                 | Reprocess stored events/logs from a checkpoint or from the beginning.                                    |
| API gateway                    | Centralizes edge concerns.                    | Route requests through one layer that handles auth, rate limits, routing, and logging.                   |
| Service discovery              | Finds available instances.                    | Register instances in DNS or a registry and have clients/load balancers resolve them.                    |
| Load balancing                 | Spreads traffic.                              | Use L4/L7 balancers with health checks and algorithms like round-robin or least-connections.             |
| Request tracing                | Follows work across services.                 | Propagate trace IDs and emit spans from every service involved.                                          |
| Versioned APIs                 | Evolves contracts safely.                     | Add `/v2`, version headers, or backward-compatible fields and migrations.                                |
| Schema evolution               | Changes data formats safely.                  | Add optional fields first, deploy readers before writers, then remove old fields later.                  |
| Webhooks                       | Notifies external systems.                    | Send signed HTTP callbacks with retries and event IDs.                                                   |
| Polling vs push                | Chooses update delivery style.                | Poll with periodic requests, or push through WebSockets, SSE, webhooks, or queues.                       |
| Stream processing              | Handles continuous events.                    | Consume from logs/topics and update state windows, counters, alerts, or projections.                     |
| Batch processing               | Handles large offline jobs.                   | Run scheduled jobs over stored data using workers, Spark, SQL, or map-reduce-style systems.              |
| Aggregation counters           | Avoids scanning raw data.                     | Increment stored counters when events happen, often with reconciliation jobs.                            |
| Approximate counting           | Scales huge cardinality questions.            | Use sketches like HyperLogLog, Bloom filters, Count-Min Sketch, or sampling.                             |
| Authn vs authz                 | Separates identity from permission.           | Authenticate users first, then check roles/policies/object ownership for actions.                        |
| Capability tokens              | Grants limited authority.                     | Issue signed tokens containing scope, resource, and expiry.                                              |
| Scoped access control          | Limits access by boundary.                    | Model permissions around tenants, orgs, projects, roles, and object-level rules.                         |
| Audit log                      | Records sensitive activity.                   | Append immutable records of actor, action, target, time, and metadata.                                   |
| Strong vs eventual consistency | Balances correctness and availability.        | Use transactions/quorums for strong consistency, or async replication/events for eventual consistency.   |



____________________
# Q&A BELOW 


Q: In an AWS-style architecture where a public internet client needs to reach an application running inside a VPC, what infrastructure components can sit between the client and the actual application server?

A: There isn't just one mandatory chain, AWS gives you several "front doors," but for a normal public web/API app in a VPC, the common shape is:
- Browser
	- [[Amazon Route 53|Route 53]] [[Domain Name Service|DNS]]
	- [[Amazon CloudFront|CloudFront]] (optional)
	- [[Amazon Web Application Firewall|AWS Web Application Firewall]] (optional)
	- Internet-facing [[Amazon Application Load Balancer|ALB]] in public subnets
	- Target group
	- [[Amazon EC2|AWS EC2]]/[[Amazon Elastic Container Service|ECS]]/[[Amazon Elastic Kubernetes Service|EKS]]

Simple public web app:
- Browser
- [[Amazon Route 53|Route 53]]
- [[Amazon Application Load Balancer|ALB]] in a public subnet
- [[Amazon Elastic Container Service|ECS]] app in a private subnet

With a CDN and WAF
- Browser
- [[Amazon Route 53|Route 53]]
- [[Amazon CloudFront|CloudFront]] and [[Web Application Firewall|WAF]]
- [[Amazon Application Load Balancer|ALB]] in public subnet
- [[Amazon Elastic Container Service|ECS]] app in private subnet

API Gateway to private app
- Browser
- [[Amazon Route 53|Route 53]]
- [[API Gateway]]
- VPC Link
- [[Amazon Application Load Balancer|ALB]]/[[Amazon Network Load Balancer|NLB]]
- [[Amazon Elastic Container Service|ECS]] app in private subnet