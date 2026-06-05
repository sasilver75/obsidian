---
aliases:
  - Concurrent
---


**Concurrency** is the ability of a program to make progress on multiple tasks during overlapping time periods, by interleaving their execution. The tasks need not actually run at the same instant — concurrency is about *structure*: composing a program as independent units of work that can be paused, resumed, and coordinated.

A single CPU core can run concurrent code by rapidly switching between tasks (e.g., waiting on I/O while another task computes).

Common mechanisms: threads, coroutines, async/await, event loops, goroutines, actors.

Contrast with parallelism, which is about literal simultaneous execution on multiple cores. Concurrent code may or may not run in parallel; parallel code is inherently concurrent.
