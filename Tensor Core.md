- A type of GPU core (along with [[CUDA Core]]s) which make up [[Streaming Multiprocessor]]s in [[GPU]]s.
	- While CUDA Cores handle scalar arithmetic operations, [[Tensor Core]]s handle matrix operations! ==They operate on entire matrices with each instruction.==
	- For example, the `mma` [[PTX]] instructions alculate D= AB + C for matrices A,B,C,D. 
- Tensor cores are much larger and less numerous than [[CUDA Core]]s. An H100 SXM5 only 4 Tensor Cores per Streaming Multiprocessor, compared to hundreds of CUDA cores.
- Tensor Cores were originally introduced in the V100 GPU, which represented a major improvement in the suitability of NVIDIA GPUs for large NN workloads. See the v100 [whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)!




![[Pasted image 20250221111145.png]]