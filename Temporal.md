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

__________

Founder timeline:
- Amazon: Maxim + Samar worked on ideas around durable workflow execution at AWS, helping to launch [[Amazon Simple Workflow Service]].
- Microsoft: Samar later created the Azure Durable Task Framework, which influenced Azure Durable Functions.
- Uber: They reunited at Uber and co-created Cadence, a workflow engine for coordinating Uber's microservices.
- 2017: Uber opensources Cadence as an orchestration engine for routing requests, coordinating microservices, and handling scalable/fault-tolerant workflows.
- 2019: Temporal Technologies founded around a fork/evolution of Cadence.

> "How do you write reliable, long-running, multi-service business logic without hand-building queues, retry tables, timers, timeout jobs, state machines, callback chains, and recovery logic?"

Temporal lets you write long-running business processes as normal code, while Temporal persists the process's progress in an append-only Event History. If workers crash, deploys happen, networks fail, or a process waits for days, Temporal can replay the workflow code from history and resume from the last durable point.

A Temporal is a strong fit for [[Saga|Sagas]] specifically, because the workflow can durably remember:
1. Which local transactions completed
2. Which [[Compensating Action]]s are registered
3. Which step is currently retrying
4. Which timeout fired
5. Whether human/manual intervention is needed

In an orchestrated saga, the hard parts are durable state, knowing which steps completed, deciding what to retry, deciding what to compensate, and recovering after the orchestrator crashes.

Temporal directly attacks those problems. The workflow history tells you which steps completed. The workflow code defines the compensation order. The service persists timers, activity results, signals, and failures. So instead of building a custom saga table, queue consumers, timeout jobs, retry workers, and repair dashboard, you encode the process in workflow code and let Temporal run it durably.

A Temporal saga looks like this conceptually:

```python
try:
      add_compensation(cancel_order_if_created)
      await create_order()

      add_compensation(release_inventory_if_reserved)
      await reserve_inventory()

      add_compensation(refund_payment_if_charged)
      await charge_payment()

      await arrange_shipping()
      await mark_confirmed()
  except Exception:
      await run_compensations_in_reverse_order()
      raise
```
Temporal says: “Write the process as code. I will persist the execution history and keep driving it until it completes.”

A few core concepts:
- ==Workflow==: Deterministic orchestration code; what should happen, and in what order.
	- Can wait for signals, start activities, call child workflows, branch, loop, handle errors.
- ==Activity==: Non-deterministic work against the outside world: HTTP calls, DB writes, payment APIs, email, inventory updates LLM calls, file I/O.
	- Activities are retryable, and should be [[Idempotency|Idempotent]] ([[At Least Once|At-Least-Once-Delivery]]).
	- Temporal records completed activity results, so replay does not redo already-completed activities.
- ==Worker==: Your process that runs workflow and activity code.
- ==Task Queue==: A durable routing/load-balancing queue polled by workers. It lets you scale worker pools horizontally and route work to the right service/language/runtime.
- ==Event History==: The source of truth for a workflow execution. Temporal doesn't restore memory snapshots, it *reruns workflow code and feeds it recorded events so that it reaches the same state.*
# KEY CONSTRAINT: Workflows must be deterministic
- No Date.now(), random number generation, DB calls, HTTP calls, thread sleeps, or filesystem reads inside workflow logic unless using replay-safe Temporal APIs. Put those operations inside [[Idempotency|Idempotent]] Activities.




