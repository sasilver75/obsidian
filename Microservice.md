---
aliases:
  - Service-Oriented Architecture
  - SOA
---

When you split a [[Monolith]] into separately-deployable services, often owned by different teams with their own databases.

The upside is independent deployment, independent scaling, clearer ownership, and fault isolation.

But the cost is very real. You introduce distributed systems concerns: [[Retry|Retries]], [[Timeout]]s, partial failures, [[Tracing]], [[Service Discovery]], API versioning, [[Eventual Consistency]], duplicated data, deployment orchestration, [[Distributed Transaction]], and harder local development.

A lot of teams adopt microservices to solve what are actually code organization problems in their microservices (see [[Modular Monolith]]), then later discover that they now have both code organization problems and distributed systems problems.

Best when:
- Teams are large enough that deployment coordination is painful
- Domains are stable and well-understood
- Services need different scaling/security/runtime characteristics
- You can afford serious platform/observability investment




