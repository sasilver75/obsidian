A probe that reports whether an instance is ready to receive traffic. Use it during startup, deployment, and dependency outages so traffic is not routed to unready instances.

> "Can this instance safely handle new requests right now?"

On failure, a (e.g.) load balancer would stop routing traffic to it.

A readiness check is about serving traffic. An application can be alive but not ready. For example, a web server process may be running, but still loading configuration, warming caches, applying migrations, waiting for a database connection pool, or temporarily overloaded. In that case, the readiness check should fail so the load balancer stops sending new requests to that instance.