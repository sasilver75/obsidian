---
aliases:
  - AWS Step Functions
---
A managed orchestration service where you define a workflow as a state machine. 
- The workflow is made up of states: `Task, Choice, Parallel, Map, Wait, Succeeed, Fail`, etc.
- A `Task` might invoke [[Amazon Lambda|Lambda]], call an AWS SDK action, wait for an external callback token, run ECS, start [[AWS Glue|Glue]], invoke [[Amazon Sagemaker|Sagemaker]], etc.
- The default choice for new AWS workflow/orchestration systems over [[Amazon Simple Workflow Service|AWS Simple Workflow Service]], which is older and lower-level.
- Has two major execution modes:
	- Standard Workflows: Durable, auditable, up to one year, suited for long-running business processes
	- Express Workflows: High-throughput, shorter-lived, up to five minutes, suited for event processing




