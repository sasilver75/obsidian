
Top-level `/services` rundown:
- `frontend`: The solidJS browser application that provides the main Omega UI and connects useres to Maestro, authentication, telemetry, and other externally exposed endpoints.
- `maestro`: The central GraphQL backend-for-frontend that stores application state in PostgreSQL and coordinates requests across the specialized backend services and Temporal workflows.
- `athena`: A Go gRPC service that provides product-specific access to the Neo4j knowledge graph, including querying/copying/editing/exporting formations, laydowns, and scenario data.
- `aegis`: The main AI-orchestration service, exposing HTTP and gRPC APIs and running Temporal workers for agent workflows like COA generation, analysis, and weaponeering.
- `argus`: An ML inference service wrapping the MADiff ISR allocation engine to predict trajectories, orbit stations, track quality, and tasking recommendations.
- `herald`: Hosts SAGE agent execution logic and SMACK-owned MCP tool implementations, other than the RAG tools implemented by Ragnarok.
- `hermes`: The MCP tool-control plane that advertises AI-safe tools and brokers their execution through Temporal workflows, without implementing the underlying tools itself.
- `hopper`: A streaming-ingestion service that reads track data from an upstream TCP feed, converts it to a structured protobuf message, and broadcasts it to gRPC subscribers.
- `iris`: A provider-neutral object-storage service exposing gRPC operations for uploads, downloads, deletion, and presigned URLs over S3, Azure Blob Storage, or local storage.
- `monsoon`: A gRPC service intended to expose the Typhoon planning model to Maestro. At the moment, it's still somewhat early-stage and scaffolded.
- `parallax`: A session-oriented synthetic-warfare simulation service supporting LLM-controlled assets, human expert takeover, knowledge-graph queriers, and CLEAR decision-data capture.
- `ragnarok`: The retrieval-augmented-generation service that extracts and chunks documents, obtains embeddings through Rosetta, stores vectors in PostgreSQL/pgvector, and retrieves relevant passages.
- `rosetta`: A shared LLM gateway that normalizes access to OpenAI, Anthropic, Bedrock, and other model transports while supporting structured output, embeddings, and MCP tool calling.
- `saucy`: The centralized authorization engine that evaluates roles and permissions and answers whether an authenticated user may perform an action.
- `simple`: A faster-than-real-time operational simulation engine for modeling military platforms, movement, sensing, detection, environmental effects, and engagements.
- `sisrs`: The SMACK ISR Scheduler (SISRS), which generates collection plans using mission requirements and available platforms, sensors, and optimization capabilities.
- `smack-sensors`: The current authentication service, providing login, user storage, signed JWT issuance and token verification for other services.
- `spicy`: The current authentication service, providing login, user storage, signed JWT issuance, and token verification for other services.
- `wizard`: The weaponeering and ISR planning service that turn scenario geometry, force laydowns, targets, and constraints into threat maps, intermediate artifacts, and optimized plans. 

Platform and deployment repositories
- `smackbackend`: Owns the umbrella Helm chart and environment-specific values used to install and update the complete SMACK application in Kubernetes .
- `infrastructure`: Contains the Pulumi IaaS code that provisions EKS/AKS clusters, namespaces, cloud databases, IAM roles, and bootstrap K8s secrets.
- `smithy`: Provides customer-specific deployment utilities, including PostgreSQL and Neo4j seed jobs and tooling for mirroring required third-party images into approved registries.
- `portier`: Packages and runs the Grobi token server used by optimization services like SISRS.
- `canary`: Performs scheduled vulnerability scans of published platform images and sends reports when Trivy, OSV-Scanner, or optionally Grype detects findings.
- `theseus`: Builds SMACK-maintained patched packages for vulnerabilities that upstream package maintainers have not yet addressed.
- `tomahawk`: Owns shared protobuf API definitions and their linting and code-generation workflows so that services can communicate through consistent gRPC contracts.
- `shuffle`: Imports source data such as Google Sheets into SQLite or PostgreSQL and can produce SQL seed dumps consumed by the application and deployment deployment tooling.

Miscellaneous:
- `18ac_model`: A research-owned Gurobi MILOP model for XVIII Airborne Corps ISR and strike planning whose production-facing functionality is inteded to be lifted into Monsoon, rather than deployed directly.

How a service runs on your laptop is configured under `modes` in `infra/user-config.json`
- `native`: Run source code directly on host.
- `local`: Build local source into a Docker container.
- `registry`: Pull a prebuilt image and run it in Docker locally.
- `off`: Do not run the service.

Importantly, `registry` doesn't mean "run in a remote environment," it still runs on your laptop, only the application image came from GitLab.

Which Neo4j instance, model providers, models, etc. is also configured under `env` in `infra/user-config.json`
```
"NEO4j_INSTANCE": "local"
```
Possible values:
- `local`: Start the Compose Neo4j container
- `dev`: Use the named Aura Dev credentials
- `demo`: Use the named Aura Demo credentials
- Empty (`""`): Generates/adds no Neo4j variables to the `infra/.env`. Services like Athena/Aegis will generally fail.
Only the literal value `"local"` starts the Compose Neo4j container. Every other name is looked up in `system-credentials.json`.

Smackspace has four local credential/configuration inputs:
- `infra/configurations/common-env.json`: Non-personal defaults: ports, local service addresses, local Neo4j password.
- `infra/user-config.json`: Your service modes (native, local, registry, off) and environment selections (Neo4j instance, default LLM models, etc.)
- `infra/credentials/user-credentials.json`: GitLab, AWS, Azure, and LLM provider credentials.
	- Note that despite "user-credentials" implying personal credentials, I got both this file and the system-credentials file from Jules via Slack, so it seems like we're all sharing the same user-credentials and system-credentials creds.
- `infra/credentials/system-credentials.json`: Shared company/service credentials: named dev/demo Neo4j instances, Postgres, Redis map tokens, server keys.
	- Q: Why is there only a single "postgres' set of credentials, while for Neo4j we have dev/demo instances? Don't we use a different postgres instance for dev/demo?"
		- A: This system-credentials file isn't a complete registry of every deployed environment's database! Local SMACK services are ordinarily expected to use one local PostgreSQL container.  Dev has its own *Aurora PostgreSQL database*, demo has a separate Aurora PostgreSQL database, and staging/gov similarly have their own. Their connection strings and passwords are provisioned by Pulumi into environment-specific K8s secrets such as `smack-store`, and not copied into your local `system-credentials.json`.


