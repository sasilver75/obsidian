---
aliases:
  - Durable Execution
---
e.g. Tools like [[Temporal]] or [[Amazon Step Functions|AWS Step Functions]]

Means running application logic in a way that can survive process crashes, deploys, machine failures, retries, and long waits without losing progress.


# Temporal's Basic Model
- Workflow: Durable orchestration logic
- Activity: Side-effecting work, like a charging a credit card, calling API, writing to DB
- Worker: Process that runs workflow/activity code
- Task Queue: The queue that workers poll for work
- Event History: A persisted log of everything that happened
- Replay: Rebuild workflow state from event history after failure

Why it matters in SD: Handles business processes that do not fit cleanly into a single request/response or into one queue job.
Good for:
- Payment flows
- Order fulfillment
- User onboarding
- Multi-step provisioning
- Data pipelines
- Human approval workflows
- Retry-heavy API integrations
- [[Saga]]s across multiple services

Gotcha: ==Workflow code usually must be deterministic, because the engine might replay it from history to reconstruct state. Side effects belong in *activities*, not directly in workflow logic.==

