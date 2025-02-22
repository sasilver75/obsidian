---
aliases:
  - GPU RAM
---

References:
- [Modal: What is GPU RAM?](https://modal.com/gpu-glossary/device-hardware/gpu-ram)

- The global memory of the GPU is a large (many MB to GBs) memory store ==addressable by all of the GPU's [[Streaming Multiprocessor]]s==
- Uses [[DRAM]] cells, which are slower but smaller than the [[SRAM]] cells used in SM caches.
- It's generally *not on the same die as the SMs*, though in the latest data center-grade GPUs like the H100 it's located on a shared interposer for decreased latency and increased bandwidth (aka "high-bandwidth memory" ([[High-Bandwidth Memory|HBM]])).
- RAM is used to implement the global memory of the [[CUDA]] programming model, and to store register data that spills from the [[Register File]]s.
- An H100 can store 80GiB in its VRAM!



![[Pasted image 20250221121931.png]]


