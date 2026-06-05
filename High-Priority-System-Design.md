
June 4: Just a note where I want to put down the terms that I really need to understand well, and then I'll use Codex to generate an Anki deck from it.


[[Consistency]] spectrum
- [[Strong Consistency]]/[[Strong Consistency|Linearizability]]
- [[Strong Read-After-Write Consistency]]
- [[Read-your-Writes Consistency]]
- [[Monotonic Reads Consistency]]
- [[Causal Consistency]]
- [[Consistent Prefix Consistency]]
- [[Eventual Consistency]]
[[Isolation]] levels
- [[Serializable Isolation]]/[[Serializable Isolation|Serializability]]
- [[Repeatable Read Isolation]]
- [[Read Committed Isolation]]
- [[Read Uncommitted Isolation]]
Read Anomalies:
- [[Dirty Read]]
- [[Non-Repeatable Read]]
- [[Phantom Read]]
- [[Write Skew]]
How can Postgres be used as a Queue? 
[[Conflict Resolution]]
[[Distributed Transaction]]s, See [[Transactional Outbox Pattern|Outbox Pattern]], for instance.
[[Change Data Capture]] (CDC) (vs Outbox)
[[Distributed Transaction]]s generally
[[Cache]] stuff
- [[Cache Write Strategy|Cache Write Strategies]] ([[Write-Through Cache|Write-Through]], [[Write-Around Cache|Write-Around]], [[Write-Back Cache|Write-Back]])
- [[Cache Read Strategy|Cache Read Strategies]] ([[Cache-Aside]], [[Read-Through Cache|Read-Through]])
- [[Cache Eviction Strategy|Cache Eviction Strategies]]
- [[Cache Warming]]
- [[Refresh-Ahead]]
- [[Stale-While-Revalidate]]
- [[Time to Live]]
- [[Cache Stampede]]
- [[Redis]] datatypes, use cases, key naming conventions, etc.
- [[In-Process Cache]]
- [[Client-Side Cache]]
[[TCP Termination]], [[TLS Termination]], when do they happen, and why? 
- HTTP keep-alive vs TCP keepalive
[[API Gateway]]s
[[Load Balancing|Load Balancer]]s
[[Webhook]]s
Realities of running [[WebSockets]] and [[Server-Sent Event|SSE]]s operationally (stateful -> failover, etc)
[[Pagination]] ([[Pagination|Cursor-Based Pagination]] vs [[Pagination|Offset-Based Pagination]])
Brief review of [[B-Tree]], [[Clustered Index]], [[Composite Index]]es
[[Geospatial Index]]es (just [[R-Tree]] is fine)
Quick review of [[Write-Ahead Log]] and its uses
Quick review of [[Replication]] strategies
- [[Single-Leader|Leader-Follower]] vs [[Multi-Leader]] vs [[Leaderless]]
- [[Synchronous Replication]] vs [[Asynchronous Replication]]
- [[Quorum]] replication vs full replication ([[Quorum Write]], [[Quorum Read]], [[Gossip]])
- [[Physical Replication]] vs [[Logical Replication]]
- Cross-regional replication (sync vs async)
Mechanics and Flow of [[JSON Web Token|JWT]], [[JSON Web Key Set|JWKS]], [[Refresh Token]]
[[User Access Token]]s vs [[Service Token]] 
Comparison with [[Session|User Session]]s stored in DB
High-level on [[OAuth]], [[OpenID Connect|OIDC]],  and how they (and anything else related to authn, auhz) is actually implemented in a SD
[[Blob Storage|Object Storage]] usage specifics (presigned url flow, key naming conventions)
[[Amazon SNS|SNS]] and [[Amazon SQS|SQS]] use together (SNS providing pub-sub fanout, while SQS providing durable queue-based message buffering)
[[Kafka]] Topics
[[Kafka Connect]]
Actual event processing, what it looks like (e.g. [[Dead Letter Queue]]s, processing one at a time or batch-wise, broker bookkeeping)
[[Materialized View]]
[[Sticky Session]]

"What happens when you go to Google.com?"

[[Golden Signals of Monitoring]]
[[Circuit Breaker]]
[[GeoDNS]], [[GeoRouting]]
[[Fallacies of Distributed Systems]]
[[Compaction]]





