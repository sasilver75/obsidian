[Site: Temporal](https://temporal.io/)

A ==durable execution platform.== It lets you write long-running, fault-tolerant workflows as ordinary  code, without manually managing ==state==, ==retries==, ==queues==, or ==failure recovery==.
- Competitors include: Conductor (Netflix OSS, one of the originals), Cadence (Direct predecessor @ Uber from Temporal founders), Inngest (newer, developer-friendly, lighter weight), Restate (newer Rust-based alternative)
- Cloud-native equivalents: [[Amazon Step Functions]] , [[Amazon Simple Workflow Service]] (SWF), [[Azure Durable Functions]], [[Google Workflows]]
- Has official SDKs for: Go (the most mature), Java, Python, TypeScript, JavaScript, .NET (C#)
	- The server is langauge agnostic; workers in different languages can even participate in the same workflow! You could have a Go workers handling some activities, and a Python worker handling others, all coordinated by the same Temporal server.

The problem it solves: Imagine a multi-step business process:
1. Charge a credit card
2. Reserve inventory
3. Send confirmation email
4. Update CRM

Any step can fail! The process might take seconds or days.
- You need retries, timeouts, and ==if something fails halfway through, you need to know where you were==.
- This is error-prone to build yourself!

Temporal lets you write this as plain code, and handles all the durability for you.

## How it works
- ==Workflows==: Ordinary functions that define the steps. Temporal automatically persists their execution state. If the process crashes mid-execution, it replays from where it left off.
- ==Activities==: The individual steps (the actual work -- API calls, DB writes). These are what gets retried on failure.
- ==Workers==: Your code that executes workflows and activities. Stateless, can scale horizontally.
- ==Temporal Server==: Stores workflow state, schedules activities, manages retries. ==You don't write to this directly==.
	- Every workflow action is logged as an event; Replay reconstructs this state by re-executing the workflow code against this history. This is why ==workflows must be deterministic;== same inputs must produce same outputs!
		- This means no use of `random.random()`, no `dateimte.now()`, no direct API calls inside workflow functions (those go into activities). This is the main footgun for new users.
	- You can run and operate this yourself, or use the managed ==Temporal Cloud== service.

```python
@workflow.defn
  class OrderWorkflow:
      @workflow.run
      async def run(self, order_id: str):
          await workflow.execute_activity(charge_card, order_id)
          await workflow.execute_activity(reserve_inventory, order_id)
          await workflow.execute_activity(send_confirmation, order_id)
```
If the worker crashes after a `charge_card` succeeds, Temporal replays the workflow and resumes at `reserve_inventory`; `charge_card` doesn't re-execute.






![[Pasted image 20260425222450.png]]