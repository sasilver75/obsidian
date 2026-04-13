Celery is a Task Queue written in Python, a piece of infra used as a mechanism to distribute work across threads or machines.
- A task queue's input is a unit of work called a tasks.
- Dedicated worker processes constantly monitor task queues for new work to perform.
- Celery communicates via messages, usually using a broker to mediate between clients and workers.
- To initiate a task, the client adds a message to the queue, the broker then delivers that message to a worker.

Celery requires a message transport to send and receive messages.
- The [[RabbitMQ]] and [[Redis]] broker transports are feature complete, but there's also feature for a myriad of other experimental solutions, like using [[SQLite]] for local development.

Can run on a single machine, on multiple machines, or even across datacenters.

Some work shouldn't happen inside a short-lived web request. Celery solves this by moving work out of the request cycle and into background worker processes.
- A web request is used to enqueue a task (fast, ms) and returns immediately.
- Later, a separate worker process picks up the task and executes it in the background, regardless of how long it takes.
- Celery also handles scheduled work, like running a task every night at 6am without any human intervention.

### Core Components

#### ==Broker==
- A message queue that sits between the code that *enqueues* tasks and the workers that *executes* them.
- When you call a task, Celery *serializes it* and drops a message in the broker (e.g. Redis).
- Workers poll the broker for messages.

### ==Worker==
- A long-running Python process that polls the broker, picks up task messages, deserializes them, and executes the task function.
- You can run many workers in parallel - each picks up tasks independently.
- Workers are stateless -- they don't share memory with eachother or with the web process -- *everything they need must come through the task arguments or the database*.


#### ==Beat== (The Scheduler)
- A separate process that *reads a schedule and enqueues tasks at the right times*.
- Beat itself does NOT execute tasks, it just drops messages in the broker on a schedule.
- Beats and workers are always separate processses.


## Tasks
- A celery task is just a Python function decorated with `@app.task`

```python
from worker.celery_app import app

@app.task
def fetch_311():
    """Fetch new 311 records from Socrata since the last run."""
    client = SocrataClient(config=DATASET_CONFIGS["sr_311"])
    result = client.fetch_incremental(since=get_high_water_mark())
    return result.rows_fetched
```

Calling `.delay()` enqueues it -- the function does not run immediately.

```python
# Enqueue the task — returns immediately
fetch_311.delay()

# The worker picks it up and runs it asynchronously
```

Task arguments are serialized to JSOn before being sent through the broker, so arguments must be JSON serializable (strings, numbers, lists, dicts, etc.)
```python
# Good — plain values
process_records.delay(dataset_id="sr_311", batch_size=1000)

# Bad — not JSON-serializable
process_records.delay(records=[<SQLAlchemy Row object>, ...])
```


### Task Chaining
- Tasks can be changed so the output of one feeds into the next, or so they run in sequence!

```python
from celery import chain

# Chain: fetch → process → refresh view
# Each step runs only if the previous succeeded
pipeline = chain(
    fetch_311.si(),         # .si() = "signature, ignore result"
    process_311.si(),
    refresh_views.si(),
)
pipeline.delay()
```
- Above: `si()` creates a "task signature" without passing the previous tasks's return value as an argument... in this case, we might use this because our tasks read from and write to the database directly, rather than passing data through the chain.

### Celery Beat: Scheduled Tasks

Beats read a schedule dict, and enqueue tasks at the right time using crontab expressions.
```python
# workers/worker/celery_app.py

from celery.schedules import crontab

app.conf.beat_schedule = {
    "fetch-311-daily": {
        "task": "worker.tasks.fetch_311.run",
        "schedule": crontab(hour=6, minute=0),  # every day at 6am UTC
    },
}
```
Above: The crontab fields are minute/hour/day_of_week/day_of_month/month_of_year.


### The Celery App object
- Everything hangs off a single `Celery` app object instance, defined once and imported everywhere.


```python
# workers/worker/celery_app.py

from celery import Celery
import os

app = Celery(
    "la_observatory",
    broker=os.environ["REDIS_URL"],
    include=[
        "worker.tasks.fetch_311",
        "worker.tasks.process_311",
    ],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
```
- Above: "include" tells Celery which modules contain task definitions; without this, the worker won't know about your tasks.


### Concurrency
- A single Celery worker process can handle multiple tasks simultaneously using either threads or subprocesses.
- In docker compose, we can use `--concurrency=2`, meaning two tasks can be run in parallel on one worker container.
```yaml
command: celery -A worker.celery_app worker --loglevel=info --concurrency=2
```
- In our LA Observatory ingest workloads, we're mostly I/O bound (HTTP fetches, DB writes), so 2 is fine. CPU-bound work would use --concurrency=1 to avoid contention.


### Flower
- `mher/flower` is an open-source web application for monitoring and managing Celery clusters, providing real-time information about the status of Celery workers, etc.



