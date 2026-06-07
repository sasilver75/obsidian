
A control mechanism where an overloaded downstream system signals upstream producers to slow down, stop sending, buffer less, retry later, or drop work.
- This prevents a slow service/queue/database from being overwhelmed by unlimited incoming work.

Generally, it works by putting a hard limit somewhere and making overload visible to the caller, instead of letting work pile up forever.
Flow:
1. Downstream system has limited capacity
2. It tracks a pressure signal: queue depth getting high, CPU/memory usage high, connection count high, request latency is up, error rate is up, etc.
3. When pressure crosses a threshold, it stops accepting unlimited work.
4. Upstream must react: slow down, retry later, reduce concurrency, buffer within a limit, shed low-priority work, or fail fast.

Example:
- A downstream API gets overloaded. Instead of accepting every request and timing out, it starts to return:
```
429 Too Many Requests
Retry-After: 5

or

503 Service Unavailable
Retry-After: 10
```
((Note: R-A:10 should usually mean "do not retry *before* 10 seconds"; clients still manage their own [[Retry|Retries]]s))
The caller then reduces concurrency, backs off, or queues only a bounded amount of work.


Message Consumers
- A worker might limit itself to 100 in-flight messages. If it already has 1010 unacked messages, it stops pulling more. This stops it from taking ownership of work it can't finish. Consumer lag then becomes the pressure signal for either autoscaling or producer throttling.

TCP
- [[Transport Control Protocol|TCP]] has built-in backpressure: If the receiver can't read fast enough, its receive buffer fills and its advertised window shrinks. The sender is forced to slow down, because the network protocol stops allowing unlimited bytes in flight.








