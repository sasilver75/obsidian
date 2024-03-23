#article 
Link: https://cameronrwolfe.substack.com/p/conditional-computation-the-birth

-------

Modern advancements in LLMs are mostly a product of scaling laws
- As we increase the size of the model, we see a smooth increase in performance, assuming that the model has sufficient amounts of data.

While scaling up these dense LLMS, we will *eventually* reach a hardware-imposed limit due to hardware memory costs and the dependence of the model's compute footprint on the total number of parameters.
- ((In other words, we don't have infinite memory, so we can't just increasingly scale parameters))

As a result, ==compute and data== have become the primary bottleneck for training better LLMs.
- We can attempt to *==avoid* this limitation by training sparsely activated language models.====

> "As the training of dense models hits the boundary on the availability and capability of the hardware resources today, ==Mixture-of-Experts== (MoE) models have become one of the most promising model architectures due to their significant training cost reduction compared to equivalent dense models!"

((Above: I'm curious why training cost is lower ... Oh, well I guess I can see why. You still have to hold it all in memory, right? You just have fewer parameters to update on the backward pass.))

[[Mixture of Experts]] layers are simple and let us increase the size/capacity of a language model without a corresponding increase in required compute!
- ==We just replace certain layers of the model with multiple copies of the layer -- called "experts" -- that have their own parameters.==
- Then, ==we use a gating mechanism to (sparsely) select the experts used to process each input==!


# TL;DR: What is a Mixture of Experts (MoE)?

The standard decoder-only transformer architecture used by most generative LLMs is shown in the figure below. 
![[Pasted image 20240322172507.png|450]]

In the context of LLMs, MoEs make a ***simple modification*** to this architecture -- ==we replace the feed-forward sub-layer with an MoE layer==!

![[Pasted image 20240322172649.png]]

==This MoE layer is comprised of several *experts* (anywhere from a few experts to thousands) where each expert is its *own* feed-forward sub-layer with an independent set of parameters!==
((We still don't know how the routing works, or how it selects 1+ options))
((So we replace the feed-forward layer with a router that routes to a subset of feed-forward layer))

==We can similarly convert encoder-decoder transformers, like T5, into MoE models by replacing feed-forward sub-layers in both the encoder and decoder with MoE layers==

In this overview, we'll primarily look at MoE models based on the [[Encoder-Decoder Architecture]]!

-----

> The ST-MoE models have 32 experts with an ==expert layer frequency== of 1/4 (meaning every fourth FFN layer is replaced by an MoE layer).

#### Sparse Expert
- The approach might *seem* problematic, because it adds a ton of parameters to the model (instead of a feed-forward layer, we've got both a router and a variety of feed-forward expert layers)... ==however we only use a small portion of each MoE layer's experts in the forward pass!==
	- As a result, ==the computational cost of an MoE model's forward pass is much less than that of a dense model *with the same number of parameters*==.

(Two) Components of an MoE Layer:
1. ==Sparse MoE Layer==
	- Replaces dense feed-forward layers in the transformer with a sparse layer of several, similarly-structured "experts", each of which is a normal feed-forward layer, like the one that the MoE layer replaced.
2. ==Router==
	- Determines which tokens in the MoE layer are sent to which experts. Has its own parameters.
	- Takes each token as input and produces a probability distribution over experts that determines to which expert each token is sent.

![[Pasted image 20240322181657.png]]
The router has its own set of parameters and is trained jointly with the rest of the network.

==Each token can be sent to many experts, but we impose sparsity by only sending a token to its top-K experts==.
- Many models set k=1 or k=2, meaning that each token is processed by either one or two experts, respectively.

Greater capacity, fixed computation
- MoE models avoid the parameter bottleneck (on memory) as we scale model sizes to enormous heights by only using a subset of the model's parameters during inference.

> "Assuming just two experts are being used per-token, the inference speed is like using a 12B model (as opposed to a 14B model), because it computes a 2x7B matrix multiplications, but with some layers shared."

Assume we have a 7B parameter LLM and replace each of its feed-forward sub-layers with an MoE layer comprised of eight experts, where two experts are activated for each token.
- This is the exact architecture that was used for Mixtral-8x7B, the MoE variant of Mistral-7B!
- The full model has 47B parameters, which all must be loaded into memory, but the model's *inference cost* is comparable to a 14B parameter model, since only 2 experts are used to process each token, leading to a ~2 x 7B matrix multiplications!
- In this way, ==we achieve the capacity of a ~50B parameter model, incurring the inference cost of only ~14B parameter models!==
	- ((When we say "capacity of ~50B", it seems like we're pretending that the experts are learning all mutually-exclusive things. I'm not sure that this is obviously true?))


==Pros and Cons==
- MoE models are widely used due to their ability to train larger models with a fixed-compute budget, but using an MoE-style LLM has both pros and cons!
	- +: Pretrain faster compared to dense models
	- +: Faster inference speed compared to dense models (with the same # parameters)
	- +: Allow us to increase model capacity, keeping the amount of compute that we use (relatively) low
	- -: Consume more VRAM, since all experts (even unused ones) need to be loaded into memory.
	- -: Prone to overfitting, and notoriously difficult to finetune, making them more complicated to use in practical applications compared to dense models.

# Origins of the Mixture-of-Experts
- Although MoEs are very popular in recent AI research, the idea has been around for a long time -- the concept of ==conditional computation== (or *dynamically turning parts of the network on/off* has its roots in work from the early 90s!)

### Early work on conditional computation
- The idea of an MoE has its origins in work done by [[Geoff Hinton]] in the early 90s, where he proposed a supervised learning framework for an *ensemble* of several networks!
- Each of these networks is expected to process a ***different subset of the training data***, and the choice of which network to use is handled by a ***gating mechanism***.
	- ((The different-subset-of-the-training-data is an interesting thing. I wonder whether that's obviously a good or bad idea.))

![[Pasted image 20240322183802.png|250]]

Several authors have explored/extended this idea since then -- with hierarchical (tree-structured) MoEs that can be trained in a supervised fashion using expectation-maximization algorithms.

![[Pasted image 20240322183832.png|250]]

Other work explored four possible techniques for estimating the gradient of stochastic neurons and hard-threshold activations functions (eg by using REINFORCE)...
These terms might seem unfamiliar, but conditional computation is one example of hard-threshold activation functions -- the activations of certain neurons are *completely eliminated* (or set to zero) from the computation. There are many computational units connected by a distributed network of gates that can be used to eliminate chunks of computation. By turning off portions of the network, we can greatly reduce the computational cost of a large NN.

Other work explored a NN setup where the nodes of the network are supplemented with gating units that determine whether a node should be calculated or not.

Other work considers MoE layers comprised of several expert networks that specialize in processing different *regions of input space* -- inputs are mapped to these experts using a *learned* gating mechanism, allowing larger networks to be computed sparsely. Going further, authors consider deeper networks with multiple, consecutive MoE layers.

Other work explores RL-based training strategies for NNs that use conditional compute -- they use policy gradient algorithms to train networks in a way that maintains accuracy while maximizing computation speed.

Other work explores ==Dynamic Capacity Networks (DCNs==), which adaptively assign capacity to different portions of the input data by defining low-capacity and high-capacity subnetworks -- for most data, low-capacity sub-networks are applied, but a gradient-based attention mechanism can be used to select portions tf the input to which high-capacity sub-networks will be applied. DCNs reduce computation and achieve performance still comparable to that of traditional convolutional neural networks (CNNs).

Deep sequential NNs discard the traditional notion of a "layer" used within NN architecture, choosing instead to construct an entire set of candidate transformations that can be selectively be applied at each layer... the selected sequence of transformations may vary highly depending on the properties of the input data.


### Outrageously Large Neural Networks: The Sparsely Gated Mixture-of-Experts Layer (Jan 2017, Shazeer)





































