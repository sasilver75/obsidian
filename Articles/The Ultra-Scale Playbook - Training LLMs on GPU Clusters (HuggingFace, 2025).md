Link: The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high_level_overview)
Video Version: [CUDA Mode Lecture 48](https://youtu.be/1E8GDR8QXKw?si=Vp7yQhS0lrCEWfPP)

What do we do when we want to train a model on multiple GPUs?
The information in this long article helps you run models on anything from 2 to 10,000 GPUs.
- Note that H100s are rentable these days for $2/hour, so for $4/hour you're doing distributed training.

-----------------------------

Training today's AI models takes thousands of GPUs humming in perfect harmony. Until recently, this was the exclusive domain of elite research labs. 

Open source has transformed the model landscape itself, but while you can download the latent LLaMA 3.1 405B, the most challenging parts are still shrouded in mystery:
- The training code
- The knowledge/technics needed to coordinate GPUs to train these massive systems

These are both shrouded in complexity and knowledge is spread around a series of disconnected papers and often private codebases.

==This open source playbook aims to walk you through the knowledge needed to scale the training of LLMs from one GPU to tens, hundreds, or thousands of GPUs!==

As the size of training clusters grew, various techniques sprung up to ==make sure that GPUs are highly utilized== at all times:
- [[Data Parallelism]]
- [[Tensor Parallelism]]
- [[Pipeline Parallelism]]
- [[Context Parallelism]]
- [[ZeRO]]
- [[Kernel Fusion]]

These techniques significantly ==reduce training time== and make the best use out of expensive hardware.

This book is built on the following ==three general foundations==:
- ==Quick Intros on Theory and Concepts==
	- How does each method work at a high level, and what are its advantages and limitations?
- ==Clear code implementations==
	- When we put code down, we discover all kinds of edge cases and important details when we implement something.
	- The [picotron](https://github.com/huggingface/picotron) repository is built for education and implements concepts in single, self-contained short files.
	- The [nanotron](https://github.com/huggingface/nanotron) repository has production-ready implementations used in training at HuggingFace.
- ==Real training==
	- Actually scaling our LLM training depends on our infrastructure, like the types of chips, interconnects, etc. -- there's no single unified recipe.
	- We ran over 4100 distributed experiments (over 16k, including test runs) with up to 512 GPUs to scan many possible distributed training layouts and model sizes.

------------

### High level overview

All the techniques we'll cover in this book tackle one or several of the ==following three key challenges==:
1. ==Memory usage==: a hard limit; if a training step doesn't fit in memory, training can't proceed.
2. ==Compute efficiency==: We want our hardware to spend the maximum amount of time computing, so we need to reduce time spent on data transfers or waiting for other GPUs to perform work.
3. ==Communication overhead==: We want to minimize communication overhead, which keeps GPUs idle. To achieve this, we'll try to make best use of intra-node (FAST) and inter-node (SLOW) bandwidths, as well as overlap communication with compute as much as possible.

In many places, we'll see that we can trade one of these (computation, communication, memory) for another.
Finding the right balance is the key to scaling training.

As this book is very extensive, we've made a cheatsheet to help you navigate it! That cheatsheet is here: [LINK](https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg)


## First Steps: Training on one GPU

Let's review the basics of model training before we scale to many GPUs.

When a model is trained on a single GPU, the training typically consists of three steps:
1. A ==forward pass== which passes inputs through the model to yield its outputs
2. A ==backward pass== to compute the gradients
3. An ==optimization step== using the gradients to update the parameters

The [[Batch Size]] is one of the most important [[Hyperparameter]]s for model training, affecting both ==convergence== and ==throughput==.

A ==small batch size== can be ==useful early in training to quickly move along the training landscape==.

However, further along in model training, smaller batch sizes ==keep gradients noisy, and the model may not be able to converge== to the most optimal final performances.

At the other extreme, a ==large batch size== gives ==very accurate gradient estimations==, but will ==result in slower convergence== (and potentially wasted compute).

For more on this, see OpenAI's 2018 paper on [Large-Batch Training](https://arxiv.org/pdf/1812.06162)

Batch size also affects the time it takes to train on a given dataset:
- ==A small batch size will require more optimizer steps to train on the same amount of samples.==
	- Optimizer steps are costly (in compute time), and the total time to train will thus increase compared to using a larger batch size.

Note: The batch size can often be adjusted quite widely around the *optimal* batch size without major impact to the performance of the model.
- In other words, the sensitivity of final model performances to the exact batch size value is usually rather low (around the optimal bs).

Note that in the LLM pretraining community, ==batch sizes are commonly reported in terms of *tokens* rather than in number of samples==. We use ==bst = [[Batch Size Tokens]]==. This makes training numbers generally independent of the exact input sequence length used during the training.

In the simplest case, either $bs$ (in samples) and $bst$ (in tokens) can be computed from the model input sequence length (seq) as follows:

$bst = bs * seq$

(From here onward, we'll talk about batch size in terms of samples, but you get it in terms of tokens by just multiplying by the sequence length)

A sweet spot for recent LLM training is typically on the order of ==4-60 million tokens per batch==.
- This has been steadily increasing over the years:
	- [[LLaMA]] was trained using a bst of ~4M tokens for 1.4T tokens
	- [[DeepSeek V3|DSV3]] was trained with a bst of ~60M tokens for 14T tokens

And as we have these larger batch sizes, we start to run into our first problem: ==out-of-memory issues!==
- What should we do when our single GPU doesn't have enough memory to hold a full batch of our target batch size?

Let's first understand how we ran out of memory:

#### Memory usage in TRansformers

When training a NN, one stores several items in memory:
1. ==Model weights==
2. ==Activations needed to compute the gradients==
3. ==Model gradients==
4. ==Optimizer states==

> You'd think that you could compute memory requirements exactly, but there's a few additional memory occupants that make it hard to be exact:
> 1. CUDA Kernels typically require 1-2 GB of GPU memory.
> 2. Some rest memory usage from buffers, intermediate results, and some memory that can't be used due to fragmentation.
> We'll neglect these because they're typically small and constant factors.

These four items are stored as tensors which come in different ==shapes== and ==precisions==.
- The shapes are determined by hyperparameters like
	- batch size
	- sequence length
	- hidden dimensions
	- number of attention heads
	- vocabulary size
	- potential model sharding
- The precision refers to formats like [[float32|FP32]], [[bfloat16|BF16]], [[float8|FP8]], which respectively require 4, 2, or 1 byte to store each single value in the tensor.
	- We'll talk more about these different precisions and their tradeoffs later in the [[Mixed Precision]] training section.
	- For now, let's just know that the memory requirements for these various formats will be different, and that will impact the memory usage of the items we need to store.

So how can we determine memory usage from these 4 variables?

One option is to just empirically measure it!

We can use the ==Pytorch Profiler== to see how memory is allocated throughout training.
- We can see that memory utilization is not a static thing, but actually varies a lot during training and during a training step:

![[Pasted image 20250302002753.png]]
See above: Forward pass, Backward pass, Optimizer step
- Anatomy of a step:
	- First, during the ==forward pass==, the memory use for activations quickly.
	- Then, during the ==backward pass==, the gradients build up; as the backward pass propagates, the stored activations used to compute the gradients are progressively cleared.
	- Finally, in the ==optimizer step==, we need all the gradients to update the parameters, and then we update the optimizer states before we start the next pass.
		- Note that we don't seem to use any memory for optimizer state for the first step
	- See that we have memory reserved for parameters  throughout the entire process.
- Interestingly, the first step looks different from the subsequent ones, why?
	- In the first step, the activations increase quickly and then plateau for a while.
	- Here, the torch cache allocator does a lot of preparation preparing memory allocations to speed up subsequent steps so that they don't require searching for free memory blogs afterwards ([explanation](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)).
	- After this first step, we see the optimizer states appearing, which generally offsets the memory usage for further training steps.

Okay, so we've got this first view of memory -- let's see how ==scaling up training is often a question of maximizing compute efficiency while keeping the memory requirements of these various items (parameters, activations, gradients, optimizer states) within the memory constraints of our GPUs.==


### Weights/Grads/Optimizer States Memory
- Let's start with the first 3 items in our list (not activations):
	- Model parameters
	- Gradients
	- Optimizer States

We can actually pretty easily estimate the amount of memory needed for them:

For a simple transformer LLM, the number of parameters is given by the following formula:

$$N = h * v + L * (12 * h^2 + 13 * h) + 2 * h$$
In that equation:
- $h$: hidden dimension
- $v$: vocabulary size
- $L$: Number of layers in the model

We can see that the term that will dominate is the $h$ term, since it's the only one growing quadratically as we scale the parameters.

Memory requirements for the parameter and gradients are simply the number of parameters multiplied by the number of bytes per parameter.
- In good old-fashioned [[float32|FP32]], training both parameters and gradients requires 4 bytes
- The optimizer, if we use [[Adam]], requires the momentum and variance to be stored, which adds another two 4-byte floats per parameter.
	- ((I think this is why we we see the memory usage for the optimizer as being twice that of the parameters, in the earlier picture))

In summary:
$$m_{params} = 4 * N$$
$$m_{grad} = 4*N$$
$$m_{opt} = (4+4) * N$$
Above: Showing the number of bytes used for parameters, gradients, and optimizer state, in an FP32 (4byte) scenario.

But what if we used lower precision?
- For stability reasons, we often don't use full precision training, but some mix of higher and lower precision called [[Mixed Precision]].
- ==The default nowadays== is to use [[bfloat16|BF16]] for most of the computations (requiring 2 bytes per parameter and gradient) as well as an additional copy of the model parameters and gradients in [[float32|FP32]] (4 bytes per parameter and gradient).
- In addition to the parameters and gradients, we need to store the optimizer states: for the Adam optimizer, this requires the momentum and the variance usually stored in FP32 for numerical stability, each using 4 bytes.

Summary (in Bytes):
$$m_{params} = 2 * N$$
$$m_{grad} = 2*N$$
$$m_{params\_FP32} = 4*N$$
$$m_{opt} = (4+4)*N$$ 
> Note: Some libraries store gradients in [[float32|FP32]] rather than [[bfloat16|BF16]]/[[float16|FP16]], which would require an additional $m_{params\_FP32}=4*N$ of memory.
> This is done for example in Nanotron, because BF16 is lossy for smaller values, and we always prioritize stability! [Link](https://github.com/deepspeedai/DeepSpeed/issues/1773)

> Note: The $m_{params\_FP32}$ is sometimes called the "master weights" in the literature and codebases.

Interestingly, mixed precision itself doesn't save memory overall, and it just distributed the memory differently across the three components (parameters, gradients, optimizer state), and in fact *ADDS* another 4 bytes over full-precision training if we accumulate gradients in FP32.

Still, computing forward/backward passes in half precision (16-bit) allows us to:
1. Use optimized lower precision operations on the GPU, which are faster
2. Reduces the activation memory requirements during the forward pass, which is a large part of the memory usage as we saw on the graph above.

![[Pasted image 20250302012042.png|800]]

For now, let's start with models that still fit in a single GPU, and take a look at the last big contributor to our memory budget: the activation memory!


#### Activation Memory












































































