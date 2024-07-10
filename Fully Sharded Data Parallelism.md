---
aliases:
  - FSDP
---
References:
- Blog: [Clika.ai/Bar Rozenman's Fully Sharded Data Parallelism (FSDP)](https://blog.clika.io/fsdp-1/)
- Blog: [Meta AI: Fully Sharded Data Parallel: Faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)

Cf: [[Model Parallelism]], [[Data Parallelism]]
Supported by PyTorch and 

A distributed training technique for large neural networks. An extension of [[Data Parallelism]] that aims to overcome memory limitations and improve training efficiency. It allows us to train orders of magnitude larger models using fewer GPUs. It's relatively free of trade-offs (compared to alternatives),  optimizing memory efficiency by sharding model parameters, gradients, and optimizer states across GPUs, and improving computational efficiency by decomposing the communication and overlapping it with both the forward and backward passes. Produces identical results as standard distributed [[Data Parallelism|Data Parallel]] (DDP) training, and is available in an easy-to-use interface that's a drop-in replacement for PyTorch's DistributedDataParallel module.

Problem: [[Data Parallelism]] distributes different batches of data across multiple GPUs, with each GPU running a full copy of the model. Forward passes and gradient are computed on each machine, then the gradients are synchronized (gathered, averaged, communicated) each epoch. Becomes problematic for large models that exceed single-GPU memory capacity. Similarly, in [[Model Parallelism]], when the first GPU in the sequence (say, in Layer-wise Model Parallelism) completes its forward pass on its batch and intends to run the forward pass on the next batch, it first has to update its gradients. This means it has to wait for the downstream GPUs to complete the forward pass and propagate their respective gradients backwards before its weights can be updated, resulting in significant idle time -- we've split out model amongst our GPUs, but our GPUs are often just sitting around waiting! [[Fully Sharded Data Parallelism|FSDP]] intends to offer a solution to this problem, allowing us to make full use of all the GPUs on large models without significant idle GPU time.

Idea: FSDP shards (splits) not *just* the data, but also the model parameters, gradients, and optimizer states across multiple GPUs. This allows for training of models that are much larger than what can fit on a single GPU.
Although the parameters are sharded to different GPUs, the computation for each microbatch of data is still local to each GPU worker. 

Sharding:
- Model parameters are split across GPUs, with each GPU storing only a ==portion of each layer== (parameters, gradients, optimizer states).
	- This "horizontal splitting" (compared to the vertical splitting of model parallelism) is often called ==Sharding==.
- During the forward pass, required parameters are gathered from other GPUs.
- After the backward pass, gradients are reduced across GPUs.
- Optimizer steps are performed locally on each GPU for *its portion of the parameters*.
- All GPUs run all the units one by one (in parallel) during forward and backward steps by gathering necessary shards of model parameters and other entities from other GPUs.

This increases the communication required between devices, due to the need to gather parameters and reduce gradients, but uses efficient communication algorithms to minimize this overhead (all-gather and reduce-scatter operations for parameter and gradient communication).

----
![[Pasted image 20240709230350.png]]

Setting the stage for FSDP
1. Split the dataset: Split the dataset into (eg) three subsets, and assign each of them to a specific GPU to be processed independently.
2. Assign units: Assign specific layers to each unit (logical vertical partition of layers of the model) that will be in charge of managing them during the training process.
3. Shard the model: Divide each entity (Parameters, Optimizer State, Gradients) into three shards and allocate them to the GPUs so that each GPU only has to hold `MEM_total/3` in its memory.

![[Pasted image 20240709224216.png]]

### Forward Pass

Broadcast model parameters: All GPUs will gather the model parameters of the first unit (MP1) so they can run the first forward step.
![[Pasted image 20240709225450.png]]
Above: Opaque colors indicate that a shard is "owned" by the GPU and will persist throughout the entire training process.

Forward pass unit 1: Each GPU will run forward pass on unit 1 on its respective batch using the complete MP 1 (ModelParameters) that each GPU gathered from all of the other GPUs.
Since each GPU has a different input batch, the activation (ACT) that each one will calculate will be different, even though all of them currently hold the same model parameters.
In some FSDP configurations, the forward pass can be performed in parallel by loading the next MP (in this case MP), which would further accelerate training... but this also increases GPU memory usage, since the GPU must hold the MP from two different units at the same time.
![[Pasted image 20240709230402.png]]


After we calculate the ACT, they're retrained in each GPU for later use in gradient computation during the backward pass.
Reshard MP 1: Delete only the broadcasted (low opacity) MP 1 from each GPU in order to free up GPU memory -- note that each GPU still holds on to the shard that was assigned to it.
![[Pasted image 20240709230409.png]]

Repeat for all other units; Repeat the process for subsequent units 2 and 3, broadcast run the forward pass, and reshard the MP while holding on to the ACT until unit 3 forward pass in done. Doing so will give us the ACT for the entire model.
![[Pasted image 20240709230418.png]]

Compute loss: For each GPU, compute the loss of its respective batch using the loss function.
![[Pasted image 20240709230425.png]]


### The Backward Pass

- Broadcast model parameters: Gather MP for the current unit -- we already have the MP at hand for the backward pass on unit 3, since we just broadcasted them to all GPUs for the forward pass. Therefore this step can be skipped for the unit 3 but is required for the backward pass of unit 2 and 1.
- Propagate backwards: Initiate backward propagation and update GRD using the ACT and MP on all GPUs for unit 3. As mentioned at the start of the forward pass section, we remark that at this point, the gradients haven't yet been calculated and are only placeholders that do not contain any actual information. In the next step, they'll be individually calculate for each GPU.
![[Pasted image 20240709230452.png]]

Accumulate gradients: Take the GRD calculated in each GPU for unit 3, sum them to get the accumulated GRD, then distribute the accumulated GRD across the GPUs. Afterwards, we reshard the broadcasted GRD 3 by removing the broadcasted GRD and replacing the existing shard of GRD in each GPU with the accumulated one ([reduce-scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html?ref=blog.clika.io#reducescatter) operation on GRD).
![[Pasted image 20240709230459.png]]

Reshard MP and ACT: Remove the broadcasted MP and ACT from all GPUs to free up GPU memory.
![[Pasted image 20240709230504.png]]

Repeat for all the other units: Repeat the previous steps, broadcast, and execute backward pass to collect GRD, and dicard ACT until the completion of backpropagation on units 2 and 1.
![[Pasted image 20240709230509.png]]

### Optimizer Step

Apply optimizer step: Run the optimizer step, update all MP and optimizer states. This constitutes a complete training step for the entire model on a single batch, achieving our goal of updating the model parameters while operating GPUs in parallel.
![[Pasted image 20240709230548.png]]
Next batch: This brings us back to the initial state but with updated MP, GRD, and OS. Now we repeat all the steps for the forward and backward propagation, as well as the optimization step, using the next batch as input until the training is complete.

----

![[Pasted image 20240710000652.png]]
> A comparison of standard data parallel training and fully sharded data parallel training. In standard data parallel training methods, a copy of the model is present on each GPU and a sequence of forward and backward passes are evaluated on only a shard of the data. After these local computations, the parameters and optimizers for each local process are shared with the other GPUs in order to calculate the global weight update. In FSDP, only a shard of the model is present on a GPU. Then, locally, all weights are gathered from the other GPUs — by means of an all-gather step — to calculate the forward pass. This gathering of weights is then performed again before the backward pass. After that backward pass, the local gradients are averaged and sharded across the GPUs by means of a reduce-scatter step, which allows each GPU to update its local weight shard.

![[Pasted image 20240710000726.png]]
