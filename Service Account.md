A service account is a ==non-human account used by an application, script, workload, or automated process to authenticate== and access systems without a person signing in.

It is commonly used for CI/CD jobs, backend services, cloud workloads, scheduled tasks, and integrations that need stable access to APIs, databases, or infrastructure.

Service accounts should be scoped to the minimum permissions required, rotated or managed through short-lived credentials where possible, and monitored like any other identity. A compromised service account can be especially dangerous because it often has persistent access and may run without direct human oversight.

Related: [[Authentication]], [[Authorization]], [[OAuth]]
