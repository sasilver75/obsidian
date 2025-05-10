References:
- [Modal Labs: What is a CUDA Core?](https://modal.com/gpu-glossary/device-hardware/cuda-core)

- A type of GPU core (along with [[Tensor Core]]s) which make up [[Streaming Multiprocessor]]s in [[GPU]]s.
- ==The CUDA Cores are GPU cores that execute scalar arithmetic instructions.==
- These are to be contrasted with [[Tensor Core]]s, which execute matrix operations.
	- Note the larger size and lower number of tensor cores.
- Unlike CPU cores, instructions issued to CUDA cores aren't generally independently scheduled. Instead, ==groups of cores are issued the same instruction simultaneously== by the [[Warp Scheduler]] but apply it to different registers. Commonly, these groups are of size 32 (a "warp"), but for contemporary GPUs, groups can contain as little as one thread (at a performance cost).
- ==The term "CUDA Core" is slightly slippery; in different SM architectures, CUDA cores can consist of different units -- a different mixture of 32-bit integer and 32bit/64bit floating point units.==
	- So when the H100 whitepaper says that an H100's GPUs SMs each have 128 "FP32 Cuda Cores," this accurately counts the number of 32-bit FP units, but is double the number of 32-bit integer or 64-bit floating point units (see diagram). For estimating performance, it's best to look directly at the number of hardware units for a given operation.

![[Pasted image 20250221111145.png]]
Above: Note the larger size and lower number of tensor cores.