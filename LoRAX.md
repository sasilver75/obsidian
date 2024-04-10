---
aliases:
  - LoRA eXchange
---
Link: https://github.com/predibase/lorax

A open-source framework that allows users to serve thousands of fine-tuned models on a single GPU, dramatically reducing the cost of serving without compromising on throughput or latency.
- Dynamic adapter loading
	- Include any fine-tuned LoRA adapter in your request, and it's loaded just-in-time without blocking concurrent requests.
- Heterogenous Continuous Batching
	- Pack requests for different adapters together into the same batch, keeping latency and throughput nearly constant with the number of concurrent adapters.
- Adapter Exchange Scheduling
	- Asynchronously prefetches and offloads adapters between GPU and CPU memory, schedules request batching to optimize the aggregate throughput of the system.
- Optimized Inference
	- High-throughput and low-latency optimizations including tensor parallelism, pre-compiled CUDA kernels ([[FlashAttention]], [[PagedAttention]], SGMV), quantization, token streaming.
- Ready for production
	- Prebuilt docker images, helm charts for K8s, prometheus metrics, and distributed tracing with Open Telemetry. OpenAI compatible API supporting multi-turn chat conversations.
- Free for commercial use
	- Apache 2.0 license -- nuff said.

![[Pasted image 20240222183907.png]]