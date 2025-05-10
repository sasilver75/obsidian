---
aliases:
  - Compute Unified Device Architecture
---
References
- [Modal GPU Glossary: CUDA (device architecture)](https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture)
- [Modal GPU Glossary: CUDA (programming model)](https://modal.com/gpu-glossary/device-software/cuda-programming-model)


Depending on the context, CUDA can refer to multiple distinct things: A high-level device ==architecture==, a ==parallel programming model== for architectures with that design, or a ==software platform== that extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in [this whitepaper](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf) (2008), which is highly recommended.


## Focusing on the *==device architecture==* part of CUDA
- The core feature of a CUDA architecture is simplicity, relative to preceding GPU architectures.
- ***Prior*** to GeForce 8800 and the Tesla datacenter GPUs that it spawned, NVIDIA GPUs were designed with a complex pipeline shader architecture that mapped softawre shader stages onto heterogenous, specialized hardware units...
	- This was challenging for the software and hardware sides alike; the required SWEs to map programs onto a fixed pipeline, and forced HWEs to guess the load ratios between pipeline steps.
![[Pasted image 20250221110603.png|900]]
- In contrast,  GPU devices with a **==unified architecture==** would be much simpler: The Hardware units are uniform, each capable of a wide array of computations. These units are known as ==[[Streaming Multiprocessor]]s (SMs)==, and their main subcomponents are the [[CUDA Core]]s and (for recent GPUs) [[Tensor Core]]s.
![[Pasted image 20250221110806.png|900]]
For an accessible introduction to the history and design of CUDA hardware architectures, see this [blog post](https://fabiensanglard.net/cuda/).
The Tesla architecture [whitepaper](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf) being introduced in 2008 was well-written and thorough...
The NVIDIA Tesla P100 whitepaper is less scholarly but documents the introduction of a number of other features that are critical for today's large-scale NN workloads, like [[NVLink]] and on-package [[High-Bandwidth Memory]].



## Focusing on the CUDA ==Programming Model==
- ==The CUDA programming model is a programming model for programming massively parallel processors.==
- There are three [key abstractions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#a-scalable-programming-model) in the CUDa programming model:
	- ==Hierarchy of thread groups==
		- Programs are executed in threads but can make references to groups of threads in a nested hierarchy, from blocks to grids.
	- ==Hierarchy of memories==
		- Thread groups have access to a memory resource for communication between threads in the group. 
		- Accessing the lowest layer of the memory hierarchy should be nearly as fast as executing an instruction.
	- ==Barrier synchronization==
		- Thread groups can coordinate execution by means of barriers.

The hierarchies of execution and memory and their mapping onto device hardware is summarized below:
![[Pasted image 20250221122930.png]]

==Together, these three abstractions encourage the expression of progress in a way that scales transparently as GPU devices scale in their parallel execution resources.==

Put provocatively, this programming model prevents programmers from writing programs for NVIDIA CUDA-architected GPUs that fail to get faster when the program's user buys a new NVIDIA GPU.

e.g.: Each thread block in a CUDA program can coordinate tightly, but coordination between blocks is limited. This ensures that blocks capture parllelizable components of the program and can be scheduled in any order. The programmer "exposes" this parallelism to the compiler and hardware. When the program is executed on a new GPU that has more scheduling units (specifically, more SMs), more of these blocks can be executed in parallel.

![[Pasted image 20250221123227.png]]

The CUDA programming model abstractions are made available to programmers as extensions to high-level CPU programming languages, like the `CUDA C++` extension of `C++`. 
The programming model is implemented in software by an instruction set architecture (ISA) called Parallel Thread eXecution, or [[PTX]], and a lowlevl assembly language (Streaming Assembler, or [[SASS]]).




