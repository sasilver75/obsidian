A deployment is the process of moving a specific version of software, configuration, or infrastructure into a target environment so that it becomes a running system. 
- It's about *how a new version is put into production:* which servers, containers, functions, routes, databases, assets, and health checks are updated, and in what order.

In contrast, a [[Release]] is when users are exposed to it.
- [[Deployment]]: The new version is installed and running somewhere.
- [[Release]]: Users are allowed to experience the new behavior.


Strategies include:
- Recreate Deployment
- [[Rolling Deployment]]
- [[Blue-Green Deployment]]
- [[Canary Release]]
- Shadow Deployment
- Ring Deployment



Typical modern deployment flow:
1. Developer merges code
2. [[Continuous Integration|CI]] runs tests, linting, type checks, security checks
3. Build system creates an artifact: container image, binary, package, static bundle, Lambda package, etc.
4. Artifact is pushed to a registry.
5. [[Continuous Integration|CD]] system deploys that exact artifact to staging or preview.
6. Smoke tests and integration tests run.
7. Database migrations run, ideally backwards-compatible.
8. Production deployment starts using ([[Rolling Deployment]], [[Blue-Green Deployment]], [[Canary Release]], etc.)
9. Health checks, logs, traces, metrics, and error rates are watched.
10. System either promotes the deployment, pauses it, rolls it back, or disables the feature flag.
