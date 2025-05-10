Resources:
- [VIDEO: Last 10 minutes or so of Lecture 16 of CS685@Amherst](https://www.youtube.com/live/cG3PQX64rKE?si=plxfQ18PaT_7Y71n&t=3118)

FlashAttention (May 2022)
FlashAttention 2 (July 2023)
FlashAttention 3 (July 2024)


![[Pasted image 20240415170254.png]]
People used to be very scared by the quadratic cost of attention
- Reformer
- SmartAttention
- etc

Tri Dao @ Stanford invented FlashAttention, where the idea is basically that you don't materialize your $N^2$ attention matrix -- instead, we just on-the-fly build small matrices and keep the statistics that we need to compute the softmax along the way. If you just compute this small part of attention matrices, the small part may be small enough that it fits in the SRAM of the GPU, which is the smallest/fastest part of GPU memory (which cannot be shared across processes, unlike High Bandwidth Memory (HBM)).


# FlashAttention v2
![[Pasted image 20240415170547.png]]
A further development of FlashAttention that was 2x faster. The idea is to have as much of the computation in (a) MatMul Flop (?).... Divisions are 4x more expensive than a multiplication (when we do attention, we have a division!) -- we want to do these *at the end*. These types of things are what FlashAttention bring -- better parallelism, causal mask, and just better work partitioning memory in the GPU. It's an incremental improvement, but still a very nice addition! 😄.


---------------------

Jeff Explanation
- Uses ==fusing of operations== (a single Kernel for matmul and softmax) and ==tiling== to improve ***memory*** efficiency from O(N^2) to O(N).
	- Tiling: Break up your two matrices that you're multiplying into tiles; do the matmul for each tile in series, so that the fragments of matrices can fit into the SMs of the GPU chip.
- Also this mathematical fact (in slideS) that we can compute attention incrementally.
