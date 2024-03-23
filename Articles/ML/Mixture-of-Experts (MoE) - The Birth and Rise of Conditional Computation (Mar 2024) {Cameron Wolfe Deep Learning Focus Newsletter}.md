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
Despite being studied for over two decades, MoEs fell short of their promise due to various technical challenges -- the authors in this paper tried to overcome some of these in this paper, applying MoEs to language modeling and translation domains.

- Prior issues with MoEs
	- GPUs are good at performing arithmetic efficiently, but ==GPUs aren't good at *branching==* (which is a major component of conditional computation)
	- Large batch sizes are needed to train NNs efficiently, but ==MoEs reduce (effective) batch sizes== (Because each expert received only some portion of the input batch, as each is routed )
	- Increased modeling capacity is most impactful when studying domains with *larger* training datasets, but MoEs in the past were studied in CV applications where there were insufficiently-large datasets.

What is a *Sparse MoE?
- ==Experts==
	- Each layer has several "experts" that are standalone neural network modules or layers with independent sets of parameters. 
	- In our situation, each expert in an MoE is a feed-forward NN with an identical architecture... but we could actually use more complex architectures! **We could even create hierarchical MoE modules by implementing each expert as its *own* MoE!**
- ==Router==
	- A parametric (and learnable) gating mechanism that selects a (sparse) group of experts used to process each

Both the experts and the gating mechanism are jointly trained along with the rest of the NN parameters via gradient descent.

==To compute the output of an MoE module, we take a weighted combination of expert outputs, where the weights are provided by the router!==

![[Pasted image 20240322220646.png|250]]

The router outputs an N-dimensional *vector* of weights (where N is the number of experts).

Although this approach might not initially seem useful, the magic happens when the router's output is *sparse!* 
- in this case, experts that receive a weight of zero are no longer considered when computing the output of an MoE! This allows us to train very large networks without significant compute requirements, as only a portion of model parameter are used at any given time.

![[Pasted image 20240322221129.png]]

Gating Mechanism
- Many different strategies have been given for routing within an MoE. 
- The simplest approach is to multiply our input by a weight matrix and apply *softmax* -- but this doesn't guarantee that the output is going to be sparse!
	- To solve this, authors propose a *modified gating mechanism* that adds sparsity and noise to this simplistic softmax gating mechanism.

![[Pasted image 20240322222050.png]]

The gating mechanism above performs routing similarly to the softmax gating mechanism, but it includes two additional steps:
1.  An adjustable amount of *Gaussian noise is added* to the output of the router prior to applying softmax.
2. All but the *output of the top-K experts are masked* (i.e. set to $-\infty$) to ensure that the selection of experts is sparse.

Balancing the experts:
- One issue with MoEs is that the network has a tendency to repeatedly utilize the same few experts during training.
- Instead of learning to use all experts uniformly, ==the gating mechanism will unfortunately converge to a state that always selects the same set of experts for every input==!
	- A ***self-fulfilling loop***: *If one expert is selected most frequently, it will be trained more rapidly, and therefore continue to be selected over the other experts!*
	- Prior work has proposed several approaches for solving the issue, but we see that experts can be balanced by adding a simple "soft" constraint to the training loss:

![[Pasted image 20240322223453.png]]
==We first define an "importance" score for each expert over a batch of input data, which can be computed by taking a sum over the gate values for each expert across the batch.==

Put simply, experts that are selected frequently in the batch will have a high importance score.
Then, we can compute an *==auxiliary loss function==* by taking the squared coefficient of variation (CV) of expert importance scores; see above.
==This loss term can be added to the model's training objective to encourage experts to receive equal importance within each batch.==

If the size of the training dataset is sufficiently small, then adding more capacity via more experts has diminishing returns, but we see that performance continues to improve up to a size of 68B parameters!

Such a finding emphasizes the synergy between MoE layers and the language modeling domain -- added model capacity is helpful given a sufficiently large training corpus.

# Applying Mixture-of-Experts to Transformers
- Now that we've studied early work on conditional computation, we can take a look at some applications of MoE to the transformer architecture, focusing on the encoder-decoder transformer architecture.

Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- After the proposal of the sparsely-gated MoE, work on transformers and language models had yet to begin using these ideas... adoption was hindered by the general complexity of MoE and training instability.
- Authors proposed an MoE-based encoder-decoder transformer architecture called the Switch Transformer that used a simplified gating mechanism to keep the training more stable, making MoE a more realistic and practical choice for language modeling applications.

When the Switch Transformer was proposed, researchers were just beginning to study neural scaling laws, which show that a language model's performance smoothly improves as we increase its size.

MoE models allow us to add another dimension to the analysis of neural scaling laws -- we can increase model size while keeping the computational complexity of the model's forward pass constant.

Applying MoE to transformers
- To create an MoE variant of an encoder-decoder transformer, we can simply convert the feed-forward sub-layers of the model into MoE layers!

![[Pasted image 20240322231239.png|450]]

- The feedforward transformation is applied in a pointwise fashion, meaning that each token is passed individually through the feed-forward network. 
	- As a result, each token in the sequence is routed to its set of corresponding experts.
	- Each token is passed through the routing function, forming a probability distribution over experts. Then, we select the top-K experts for each individual token -- tokens in the same sequence aren't always sent to the same experts.

Better Routing
- In prior work, the minimum number of active experts in any MoE layer was two.
- In the Switch Transformer, authors propose routing each token to only a *single expert* -- this is called a ==Switch Layer==
	- By routing to a single expert, we simplify the routing function, reduce computational overhead, and lessen communication costs while improving model performance.
	- The routing function used by the Switch Transformer is just a softmax gating mechanism; we pass each token through a linear layer that produces an output of size N (the number of experts), then apply a softmax transformation to convert this output into a probability distribution over experts.

![[Pasted image 20240322232344.png|250]]
Above:  The routing function

![[Pasted image 20240322232357.png|250]]


Above: The computation of the output of the MoE layer

From here, we compute the output of the switch layer by:
1. Selecting a single expert
2. Scaling the output of this expert by the probability assigned to the expert by the routing function.

In other words, we still compute a probability output by scaling the output of the selected expert by its probability; see as above. This approach allows us to train the MoE model when K=1.

Simple load balancing
- Authors employ multiple auxiliary loss functions to balance importance scores and perform load balancing between experts (i.e. meaning that each expert is sent a roughly equal number of tokens from the batch).
- We see that both of these objectives can be achieved with a single auxiliary loss function that is applied at each switch layer in the model

![[Pasted image 20240322233625.png]]

This loss is differentiable with respect to P and can be easily incorporated into training, encourages both the fraction of tokens allocated to each each expert to be $\dfrac{1}{N}$ , meaning that experts are equally important and receive a balanced number of tokens.


Capacity Factor
- We set a global "==expert capacity==" variable that determines the number of tokens that can be routed to each expert in any MoE layer.
- The equation for expert capacity is shown below:

$expert\ capacity = \dfrac{tokens\ per\ batch}{number\ of\ experts} * capacity\ factor$ 

In a switch layer, each token is routed to the expert that is assigned the highest probability by the routing mechanism.
- If too many tokens are sent to a single expert, computation for these tokens will be "==skipped==" -- these "dropped" tokens are passed directly to the next layer via the residual connection. Setting the capacity factor greater than one allows the MoE to handle cases where tokens are not perfectly balanced across experts.

These "dropped" tokens are passed directly to the next layer via the residual connection.

Setting the capacity factor greater than one allows the MoE to handle cases where tokens are not perfectly balanced across experts.

![[Pasted image 20240322235304.png|500]]


The expert 

































