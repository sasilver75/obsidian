---
aliases:
  - SM
---
References:
- [Modal GPU Glossary: Streaming Multiprocessors](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor))


- When we program GPUs, we produce sequences of instructions for its Streaming Multiprocessors to carry out!
- ==SMs of NVIDIA GPUs are roughly analagous to the cores of CPUs==
	- ==SMs both execute computations and store state available for computations in registers, with associated caches.==
	- ==Compared to CPU cores, GPU SMs are simple, weak processors==. Execution in SMs is pipelined within an instruction (like in all CPUs since the 90s) but there is no speculative execution or instruction pointer prediction (unlike all contemporary high-performance CPUs)
	- ==However, GPU SMs can execute more threads in parallel.==
		- An AMD EPYC 9965 CPU draws 500W and has 192 cores, each of which can execute instructions for at most 2 threads at a time, for a total of 384 threads in parallel, running at about 1.25W per thread.
		- An H100 SXM GPU draws 700W and has 132SMs, each of which have four "Warp Schedulers" that can each issue instructions to 32 threads (aka a warp) in parallel, for a total of 128x132 = >16,000 parallel threads running at about 5cW apiece.
			- Note too that this is truly ==parallel==- each of the 16,000 threads can make progress with each clock cycle!
			- But note too that SMs also support a large number of ==concurrent== threads, where instructions are interleaved.
				- A single SM on an H100 can ***concurrently*** execute up to 2048 threads split over 64 thread groups of 32 threads each. With 132 SMs, that's a total of over 250,000 concurrent threads.
			- CPUs can also run many threads concurrently, but switches between warps happen at the speed of a single clock cycle (over 1000x faster than context switches on a CPU), again powered by the sM's Warp Schedulers. The volume of available warps and the speed of warp switches help ==hide latency== caused by memory reads, thread synchronization, or other expensive instructions, ensuring that the compute resources (especially CUDA Cores and Tensor Cores) are well-utilized.
			- This ==latency-hiding== is the secret to GPU strength! CPUs seek to hide latency from end-users by maintaining large, hardware-managed caches and sophisticated instruction prediction. This extra hardware limits the fraction of their silicon area, power, and heat budgets that CPUs can allocate to computation. The result is much higher ==throughput== for things like NN inference.



### What is a GPU Core?
- The cores are the primary compute units that make up [[Streaming Multiprocessor]]s (SMs).
- Examples of GPU core types include [[CUDA Core]]s and [[Tensor Core]]s.
- GPU cores are comparable to CPU cores ins that they're the component that effects actual computations, but this analogy can be misleading. The larger SMs themselves are closer to being the equivalent of CPU cores.


## What is a Special Function Unit (SFU)?
- [[Special Function Unit]]s (SFUs) in Streaming Multiprocessors (SMs) accelerate certain arithmetic operations.
- Notable ones for NN workloads are transcendental math operations like `exp`, `sin`, and `cos`.)
- (Shown in pink, along with the LSUs)

### What is a Load/Store Unit (LSU)?
- A [[Load Store Unit]] (LSU) dispatches requests to load or store data to the memory subsystems of the GPU.
- Most importantly for CUDA programmers, they interact with the Streaming Multiprocessor's on-chip [[SRAM]] L1 data cache and the off-chip, on-device global RAM that respectively implement the lowest and highest levels of the memory hierarchy in the CUDA programming model.
- (Shown in pink, along with the SFUs)

### What is a Warp Scheduler?
- A [[Warp Scheduler]] of the [[Streaming Multiprocessor|SM]] decides which group of threads to execute.
- These groups of threads, known as ==warps==, are switched out on a per-clock-cycle basis, roughly one nanosecond.
	- CPU thread context switches, on the other hand, take a few hundred to a few thousand clock cycles -- more like a microsecond than a nanosecond, due to the need to save the context of one thread and restore the context of another.
	- On a GPU, because each thread has its own private registers allocated from the register file of the SM, context switches on the GPU don't require any data movement to save or restore contexts.
- Because the [[L1 Cache]]s on GPUs can be entirely programmer-managed and are shared between the warps scheduled together onto an SM, context switches on the GPU have much less impact on cache hit rates.
- (Shown in "Orange" (grey))

### What is a Register File?
- A [[Register File]] of the SM stores bits in between their manipulation by the cores.
- Register files are split into 32-bit registers that can be dynamically reallocated between different data types, like 32bit integers, 64bit floats, and (pairs of) 16-bit floating point numbers.
- Allocation of registers in a Streaming Multiprocess to threads is usually managed by a compiler like [[nvcc]], which optimizes register usage by thread blocks.
- (Shown in Blue)

### What is the L1 Data Cache?
- The [[L1 Cache]] on a SM is the private memory of the Streaming Multiprocessor.
- Each SM partitions that memory among a group of threads scheduled onto it.
- The L1 data cache is colocated with and is nearly as fast as components that effect computations (e.g. the CUDA Cores)
- It is implemented with [[SRAM]], the same basic semiconductor cell used in CPU caches and registers and in the memory subsystem of things like [[Groq]] LPUs. The L1 data cache is accessed by the [[Load Store Unit]]s of teh SM.
- CPUs also maintain an L1 cache, which is fully hardware managed. In GPUs, that cache is mostly programmer-managed.
- Each L1 cache in each of an H100 SMs can store 256KiB. Across 132 SMs in an H100 SXM 5, that's 33MiB of cache space.
	- (kibibyte and mebibyte)

### What is a Streaming Multiprocessor Architecture?
- SMs are versioned with a particular "architecture" that defines their compatibility with Streaming Assembler ([[SASS]]) code.
	- For example, an H100 seems to have the "Hopper" SM90 architecture.
- Note that most SM versions have two components: A Major version and a Minor version.
	- The major version is almost synonymous with GPU architecture family -- for example, all SM versions 6.x are of the Pascal Architecture.

### What is a Texture Processing Cluster (TPC)?
- Generally synonymous with a "pair of SMs" -- rarely encountered in contemporary discussions of GPUs, and not mapped onto a level of the CUDA programming modle's memory hierarchy or thread hierarchy, unlike Graphics/GPU Processing Clusters.

### Wait, what's a Graphics/GPU Processing Cluster (GPC)?
- A GPC is a collection of TPCs (themselves groups of SMs) plus a raster engine.
- Apparently some people use NVIDIA GPUs for Graphics ;), for which this raster engine is important.


![[Pasted image 20250221111145.png]]
((Above: Note thatI think these four "blocks" that make up this specific SM are referred to as "==Processing Blocks=="))


![[Pasted image 20250221112000.png]]



![[Pasted image 20250221120334.png]]
