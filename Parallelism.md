**Parallelism** is the simultaneous execution of multiple computations, literally at the same instant, typically across multiple CPU cores, GPUs, or machines. It is about *execution*, not structure.

Common forms:
- **Data parallelism**: the same operation applied to many pieces of data at once (SIMD, GPU kernels, `map` over a partitioned dataset).
- **Task parallelism**: different operations running simultaneously on different workers.

Requires hardware that can actually run things at the same time. Contrast with concurrency, which is about composing independently-progressing tasks regardless of whether they execute simultaneously. Rob Pike's framing: concurrency is about *dealing with* many things at once; parallelism is about *doing* many things at once.
