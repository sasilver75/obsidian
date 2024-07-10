---
aliases:
  - Layer-wise Model Parallelism
---

Cf: [[Data Parallelism]], [[Fully Sharded Data Parallelism]], [[Tensor Parallelism]]

A distributed training strategy in which different parts of the model architecture are distributed across multiple devices, with each device responsible for computing only its assigned portion.

The model is split into segments, with each assigned to a different device (eg GPU).  Different layers are on different devices ("vertical" model-parallelism; imagine drawing vertical dividing lines across a horizontal network architecture) with each device storing and computing its part of the model.

+: Enables training of very large models who wouldn't fit on a single device.
-: Can lead to device underutilization (since different parts of the model may have varying computational requirements). May introduce pipeline bubbles (idle time) in a naive implementation. More complex to implement than data parallelism. During both inference and backpropagation, outputs and gradients must be passed from sequential layer to sequential layer.
-: When the first GPU in the sequence (say, in Layer-wise Model Parallelism) completes its forward pass on its batch and intends to run the forward pass on the next batch, it first has to update its gradients. This means it has to wait for the downstream GPUs to complete the forward pass and propagate their respective gradients backwards before its weights can be updated, resulting in significant idle time -- we've split out model amongst our GPUs, but our GPUs are often just sitting around waiting! [[Fully Sharded Data Parallelism|FSDP]] intends to offer a solution to this problem, allowing us to make full use of all the GPUs on large models without significant idle GPU time.



![[Pasted image 20240415143740.png]]
((Ah, this diagram is actually correct, I think, but it's confusing. For example in the top-left image, this doesn't mean that all of the volume of the large blue square (representing all parameters) is being divided amongst the cores such that each core has a fraction; Instead, each of these small squares is supposed to be "the whole set of model weights". As evidence, in Data Parallelism in the bottom left, the "whole square" represents all of the data, and it's shown as shared/split over all of the cores. In the picture to the right of that, we're not saying that the data is being split into all these cores -- rather, each core gets a full copy of the data (which is represented by this blue square, through shrunked.)))