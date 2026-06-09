
The latency experienced by the *slowest requests*, usually measured at high percentiles like `p95` or `p99`.

It matters because users often notice the slow outliers, especially in systems that make many parallel fan-out calls (where even if average latency is low, high tail latency from one of the requests can make an app feel real unreliable or sluggish).

Example:
- If `p99` latency is 900ms, then 99% of requests finish in 900ms or less, and the slowest 1% take longer.



