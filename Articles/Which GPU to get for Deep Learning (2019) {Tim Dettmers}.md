#article
https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/

How do GPUs work?
* GPUs are so fast because they are efficient for matrix multiplication -- but why is this so? The real reason is actually memory bandwidth, and not necessarily parallelism!
* CPUs are LATENCY OPTIMIZED whereas GPUs are BANDWIDTH OPTIMIZED!
    * Visualize this as the CPU being a Ferrari and the GPU being a big truck. Say the goal is to move packages from A to B.
    * The CPU (Ferrari) can fetch some memory (packages) in your RAM quickly, whereas the GPU (big truck) is slower in doing that (much higher latency).
        * However, the CPU (Ferrari) needs to make many more trips to do its job, whereas the GPU can fetch much more memory at once.
    * In other words, the CPU is good at fetching small amounts of memory quickly, whereas the GPU is good at fetching large amounts of memory.
	    * The best CPUs have ~50 GBps, whereas the best GPUs have 750 GBps.
	    * So the more memory your computational operations require, the more significant the advantage of GPUs over CPUs. But there is still the latency that might hurt performance in the case of the GPU -- while a big truck might be able to pick up many packages with each tour, the problem is that you're waiting a long time until the next set of packages arrives.

If you ask a big truck to make many tours to fetch packages, you'll always wait for a long time for the next load of packages once the truck has departed to do the next tour -- the truck holds a lot, but its trips are slow!

However, if you now use a fleet of either Ferraris and big trucks (thread parallelism), and you have a job with many packages (large chunks of memory like matrices), then you'll be waiting for the first truck a bit, but after that you'll have no waiting time at all! Unloading the packages takes so much time that all the trucks will queue in unloading location B so that you always have direct access to your packages (memory). ==This effectively hides the latency problem==, allowing GPUs to provide the best memory bandwidth while having 
almost no drawbacks due to latency, thanks to thread parallelism.

1. So the first step is where the memory is fetched from the main memory (RAM) to the local memory on the chip (L1 cache and registers).
2. The second step is less critical to performance but still ads to the lead for GPUs: All computation that ever is executed happens in registers which are directly attached to the execution unit (a core for CPUs, a stream processor for GPUs). Usually, you want to have the fast L1 and register memory very close to the execution engine, and you want to keep these memories small so that access is fast.
	- Increasing the distance between the execution engine and L1/register memory makes the whole thing slower.
	- You want to keep these memories small so that access is fast; the larger the memory it is, then, in turn, it gets slower to access its memory (think: Finding what you want to buy in a small store is faster than finding what you want to buy in a huge store, even when you know where an item is)
	- The advantage of the GPU is that it can have a small pack of registers for every processing unit (==stream processor, or SM==), of which it has many. Thus we can have in total a lot of register memory, which is very small and thus very fast. This leads to the aggregate GPU registers size being more than 30 times larger compared to CPU, and still twice as fast!

In order of importance:
1. High bandwidth main memory
2. Hiding memory access latency under thread-parallelism
3. Large and fast register and L1 memory, which is easily programmable

The most important GPU specs for Deep Learning Processing Speed:
Let's cover some components in order of importance:
1. Tensor Cores
2. Memory bandwidth of a GPU
3. Cache hierarchy
4. FLOPs of the GPU

Tensor Cores
- ==Tensor Cores== are tiny cores that perform very efficient matrix multiplication. Since the most expensive part of any DNN is matrix multiplication, Tensor Cores are very useful! The author does not recommend any GPUs that do not have Tensor Cores.
	- a memory block in shared memory is often referred to as a ==memory tile== or a ==tile==....  
	- Tensor cores are so fast that they're actuallly often bottlenecked by memory bandwidth! Global memory is by far the largest cycle cost for matrix multiplication with tensor cores.
	- The larger the matrices, the better, for Tensor Cores

Memory Bandwidth
- We saw in the previous section that tensor cores are so fast that they are idle most of the time, as they wait for memory to arrive from global memory.
	- GPT-3-sized training results in Tensor Core TFLOPS utilization of only about 45-65%.
- This means that when comparing two GPUs each with tensor cores, one of the single best indicators for each GPU's performance is its memory bandwidth!

L2 Cache / Shared Memory / L1 Cache / Registers
- Since memory transfers to Tensor Cores are the limiting factor in performance, we're looking for other GPU attributes that can help *enable* faster memory transfer to our Tensor Cores in our GPU!
- L2 cache, shared memory, L1 Cache, and the amount of registers used are all related.
	- To perform matrix multiplication, we exploit the memory hierarchy of a GPU that goes from slow global memory, to faster L2 memory, to fast local shared memory, to lightning-fast registers -- but the faster the memory, the smaller it is!
- L1 and L2 caches are logically the same, but the L2 cache is larger and thus the average physical distance that needs to be traversed to retrieve a cache line is larger.
	- Think of the L1 and L2 caches as organized warehouses where you want to retrieve an item; even if you know where an item is, it takes longer to get the item in the larger warehouse (L2 cache). Large=slow;Small=fast;
- For Matrix multiplication, we use this hierarchical separate into smaller and smaller (thus faster and faster) chunks of memory to perform very fast matrix multiplications. 
	- We chunk the big matrix multiplications into *smaller* sub-matrix multiplications are called ==memory tiles==, or often just ==tiles==!
	- We perform matrix multiplication across these smaller tiles in local shared memory that's fast and close to the streaming multiprocessor (SM) (equivalent of a CPU core). With Tensor Cores, we go a step further -- We take each tile and load a part of each tile into the Tensor Core which is directly addressed by registers.
	- Larger tile sizes mean that we can reuse more memory. Each tile size is determined by how much memory we have per SM, and how much L2 cache we have across all SMs.

































