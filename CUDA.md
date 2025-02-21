---
aliases:
  - Compute Unified Device Architecture
---
References
- [Modal GPU Glossary: CUDA](https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture)


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






