---
aliases:
  - Data Parallel
  - Distributed Data Paralellism
  - DDP
---

Cf: [[Model Parallelism]], [[Fully Sharded Data Parallelism]]

A distributed training strategy in which multiple copies of the entire model are run on different devices (eg GPUs), each processing a different subset of the training data.

Input data is split into batches and distributed across devices.
Each device compute forward and backward passes independently.
Gradients are synchronized (usually averaged) across devices after each batch. Model parameters are updated identically on all devices. ((This basically seems like effectively increasing your batch size, since you only "learn" once you've done N batches across N devices and averaged/synchronized gradients))

+: Straightforward to implement and scales with the amount of training data.
-: Only works if your model fits on a single device, and there's communication overhead for gradient synchronization (at the same time, it doesn't require multiple communications in series for forward and backward passes like in model parallelism).

![[Pasted image 20240415143738.png]]
((Ah, this diagram is actually correct, I think, but it's confusing. For example in the top-left image, this doesn't mean that all of the volume of the large blue square (representing all parameters) is being divided amongst the cores such that each core has a fraction; Instead, each of these small squares is supposed to be "the whole set of model weights". As evidence, in Data Parallelism in the bottom left, the "whole square" represents all of the data, and it's shown as shared/split over all of the cores. In the picture to the right of that, we're not saying that the data is being split into all these cores -- rather, each core gets a full copy of the data (which is represented by this blue square, through shrunked.)))