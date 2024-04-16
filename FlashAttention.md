![[Pasted image 20240415170254.png]]
People used to be very scared by the quadratic cost of attention
- Reformer
- SmartAttention
- etc

Tri Dao @ Stanford invented FlashAttention, where the idea is basically that you don't materialize your $N^2$ attention matrix -- instead, we just on-the-fly build small matrices and keep the statistics that we need to compute the softmax along the way. If you just compute this small part of attention matrices, the small part may be small enough that it fits in the SRAM of the GPU, which is the smallest/fastest part of GPU memory (which cannot be shared across processes, unlike High Bandwidth Memory (HBM)).


# FlashAttention v2
![[Pasted image 20240415170547.png]]
A further development of FlashAttention that was 2x faster. The idea is to have as much of the computation in (a) MatMul Flop (?).... Divisions are 4x more expensive than a multiplication (when we do attention, we have a division!) -- we want to do these *at the end*. These types of things are what FlashAttention bring -- better parallelism, causal mask, and just better work partitioning memory in the GPU. It's an incremental improvement, but still a very nice addition! 😄.


