 A probe that reports whether a service or instance is healthy enough for operational use. Use it for monitoring and automation, but make sure it reflects meaningful dependency and runtime state. The umbrella term.

> "Is this thing okay?"

The typical action on failure depends on context.
- See [[Readiness Check]] and [[Liveness Check]]

A health check is generic. It might mean a shallow check like “HTTP server responds with 200 OK,” or a deep check like “database connection, cache connection, message queue connection, disk space, and dependency latency are all acceptable.” The term has no single universal behavior unless a platform defines one.