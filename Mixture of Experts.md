---
aliases:
  - MoE
---
January 23, 2017 (5 months before *Attention is all you need*)
[[Google Research]] - Including [[Noam Shazeer]], [[Quoc Le]], [[Jeff Dean]], [[Geoff Hinton]]
Paper: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
#zotero 
Takeaway: They introduce an MoE layer to the stacked LSTM context, where an MoE layer consists of a gating network and a collection of experts. The gating network outputs a *sparse* vector, selecting k of the experts to activate during the forward pass for this token. There are some questions about how to balance the gating networks and how to distribute experts across machines.

Notes:
- When someone says "8x7B"

> "'You at AI2 do not have the engineering throughput to deal with the headaches of getting mixture-of-experts to work' is a comment from my big lab friends." - Nathan Lambert


References:
- [Cameron Wolfe: Mixture of Experts (MoE) LLMs](https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)


---

Notes:
- Challenges of conditional computation (in which a subset of the neurons are active, on a forward pass):
	1. Modern computing devices (especially GPUs) are much faster at arithmetic than branching... Most works recognize this and propose turning on/off large chunks of the network with each gating decision.
	2. ==Large batch sizes are critical for performance, but conditional computation reduces the batch sizes for the conditionally active chunks of the network==.
	3. Network bandwidth can be a bottleneck in a multi-GPU setup where we need to send embeddings around the network.
	4. Depending on the scheme, loss terms might be necessary to achieve the desired level of sparsity per-chunk and/or per example. These issues affect model quality and load balancing.
	5. Model capacity is most critical for very large datasets; existing literature on conditional computation deals with relatively small datasets of up to 600k, though.
- Authors apply an MoE convolutionally between stacked LSTM layers. The MoE is called once for each position in the text, selecting a potentially different combinations of experts at each position. The different experts ==tend to become highly specialized based on syntax and semantics.== ((I believe this is still sort of disputed, on the semantics front)).
- Authors note that the idea of using multiple MoEs with their own gating network as part of a deep model is a more powerful and expressive idea than "top-level" mixture of experts, where the mixture of experts is the whole model.
- MoE layers consist of a set of n "expert networks", and a "gating network," whose output is a *sparse, n-dimensional vector.* The experts themselves are neural networks, each with their own parameters. The output is as follows:
	- ![[Pasted image 20240505113105.png|100]]
	- We save computation given the *sparsity* of the output of G(x); wherever G(x)\_i = 0, we don't need to compute E_i(x).
	- The authors have up to thousands of experts (per layer? per network?) but only need to evaluate a handful of them for every example. If the number of experts were even larger, we could reduce the branching factor by using a two-level hierarchical MoE.
- Softmax Gating
	- Some other papers use a non-sparse gating function, where we re-term our Gating function to output non-sparse vectors by multiplying the input by a trainable weight matrix W_g and then apply the softmax function:
		- ![[Pasted image 20240505113738.png|150]]
- Noisy Top-K Gating
	- We add two components to the Softmax gating network: *sparsity* and *noise*. Before taking the softmax function, we add tunable Gaussian noise, and then keep only the top-k values, setting the rest to -INF (which causes the corresponding gate values to equal 0). The sparsity serves to save computation, as described above. The amount of noise per component is controlled by a second trainable weight matrix W_noise:
	- ![[Pasted image 20240505114013.png|300]]
	- Where the H(x) is whatever transformation of the input our gating network uses (eg FFNN) plus some gaussian noise. This noise is useful for load balancing, as we'll discuss later.
- Shrinking Batch Problem
	- On modern CPUs/GPUs, large batch sizes are needed for computational efficiency, so as to amortize the overhead of parameter loads and updates. If the gating network chooses k out of n experts for each example, then for a batch of b examples, each expert receives a much smaller batch of approximately $kb/n << b$ examples. This causes a naive MoE implementation to become very inefficient as the number of experts increases.
- Balancing Expert Utilization
	- We observed that the gating network tends to converge to a state where it *always* produces large weights for the same few experts.
	- We take a soft constraint approach, defining the importance of an expert relative to a batch of training examples to be the batchwise sum of the gate values for that expert.
	- ==We define an additional L_importance loss which is added to the overall loss function for the model. This loss is equal to the square of the coefficient of variation of the set of importance values, multiplied by a hand-tuned scaling factor w_importance.==
	- This encourages all experts to have equal importance.
	- ![[Pasted image 20240505115135.png|300]]
	- This loss function can ensure equal importance, but experts can still receive very different numbers of examples... to solve this, we introduce a second loss function, L_load, which ensures balanced loads.

Related work (roughly chronological):
- Earlier: Hinton/Jordan's "Adaptive Mixtures of Local Experts"
- Future:
	- [[GShard]]
	- [[Switch Transformer]]
	- [[GLaM]]

Abstract
> The capacity of a neural network to absorb information is limited by its number of parameters. ==Conditional computation==, where ==parts of the network are active on a per-example basis==, has been proposed in theory ==as a way of dramatically increasing model capacity without a proportional increase in computation==. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, ==achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency== on modern GPU clusters. We introduce a ==Sparsely-Gated Mixture-of-Experts layer (MoE)==, consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.


# Paper Figures
![[Pasted image 20240505111024.png]]
Above: Because this paper came out before the Transformer, this is shown in the context of RNNs. But the MoE block is the same (as far as I can tell) as ones we might use in a Transformer, though the Transformer MoE blocks often will have a FFNN layer after combining results from experts.

# Non-Paper Figures



------

From Cameron Wolfe's [Mixture of Experts Article](https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)

In an area of study that's rapidly changing, the decoder-only transformer architecture has remained one of the few enduring staples in large language model...

MoE-based LLMs introduce sparsity to the models architecture, ==allowing us to increase its size (in terms of the number of total parameters)== ==without a corresponding increase in compute costs.==

The relevant papers here (among others) are:
- The Sparsely-Gated Mixture of Experts Layer
- Switch Transformers
- Stable and Transferable Mixture-of-Experts (ST-MoE)
- DeepSeek MoE

What are experts?
- In the standard architecture, we have a SINGLE feed-forward NN component, usually made up of ==two feed-forward layers with a non-linear activation in between.
- An MoE slightly modifies this block architecture -- instead of having a single feed-forward network within the feed-forward component of the block, we create ==several== feed-forward networks, each with their own independent weights. We refer each of these networks as an "expert."
- For example, an MoE-based LLM may have eight independent experts in each of its feed-forward sub-layers:
$Experts: {E_i(\cdot)}^N_{i=1}$ 
We can refer to each i'th expert using the notation $E_i$.

Creating an MoE-based tarnsformer:
- We simply convert the transformer's feed-forward layers to MoE, or *expert* layers.
- ==Each expert in the MoE layer has an architecture that is identical to the original FFNN from that layer.==

![[Pasted image 20250128143105.png|500]]

However we need not use experts for for *every* FF layer in the transformer.

Most MoE-based LLMs use a ==stride of P, meaning that every P'th layer is converted into an expert layer, and others layers are left untouched.== These are also called ==interleaved MoE layers==.
This approach can be used to achieve a better balance between the resulting model's performance and efficiency.

### Routing algorithms:
- The primary benefit of MoE-based architectures is their efficiency, but what do we do now that we have multiple experts? We want to ==sparsely select== the experts that should be used in each layer.
- ==Our goal is to select a subset of experts to process each token.==
	- In MoE literature, we usually say that the token will be *routed* to these experts! 
- The simplest routing algorithm would apply a linear transformation to the token vector, forming a vector of size N (i.e. the number of experts).
	- Then, we can apply a [[Softmax]] function to form a probability distribution over the set of experts for our token.
	- We can use this distribution to choose experts to which our token should be routed by simply ==selecting the top-L experts in the distribution==.

![[Pasted image 20250128144006.png|400]]

This routing strategy was used in the original Shazeer MoE paper which proposed to sparse MoE layer structure that we use today. But ==such a routing mechanism doesn't explicitly encourage a balanced selection of experts!==
- For this reason, the model is ==likely to converge to a state of repeatedly selecting the same few experts for every token instead of fully and uniformly utilizing its expert layers!== This is commonly referred to as [[Routing Collapse]]!

> The gating network tends to converge to a state where ti always produces large weights for the same few experts. This imbalance is *self-reinforcing*, as favored experts are trained more rapidly and thus selected even *more* by the gated network.

Active Parameters:
- Because we only select a subset of experts to process each token within an MoE layer, there is a concept of =="active" parameters== in the MoE literature.
	- Only a small portion of the MoE model's total parameters are active when processing a given token. 
	- ==As a result, the total computation performed by the MoE is proportional to the number of active parameters, rather than the total number of parameters.==

### Auxiliary Losses and Expert Load Balancing
- In order to encourage a balanced selection of experts during training and avoid [[Routing Collapse]], we can simply ==add an additional constraint to the training loss that rewards the model for uniformly leveraging each of its experts==!
- We define an ==importance score== for each expert!
	- This importance score is based on the probability predicted for each expert by the routing mechanism.

![[Pasted image 20250128144641.png]]

1. ==Given a batch of data, we compute importance by taking taking a sum of the probabilities assigned to each expert across all tokens in the batch.==
2. Then, to determine if these probabilities are balanced, by take the squared [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) (CV) of the expert importance scores. ==Put simply, the CV will be a small value if all experts have similar importance scores, and vice versa== .
3. From here, ==we can simply add the importance loss shown above to our standard language modeling loss to form our new objective==, which helps to ensure that the MoE assigns equal probability to experts throughout the training process.

Load Balancing
- But just because experts are assigned equal equal importance doesn't mean that tokens are routed uniformly!
	- Currently, experts would have equal importance with:
		- A few tokens that assign them very high probability
		- A much larger number of tokens that assign lower probability
	- As a result, the number of tokens dispatched to each expert can still be highly non-uniform even when using an importance loss, which can lead to excessive memory usage and generally-degraded efficiency for the MoE...
----
Sam Aside:
I think he means a situation where:

```
Expert A:
- 10 tokens each assign 0.9 probability to Expert A
- 90 tokens each assign 0.0 probability to Expert A
- Total importance ≈ 9.0 (10 × 0.9 + 90 × 0.0)

Expert B:
- Those same 10 tokens each assign 0.1 probability to Expert B
- Those same 90 tokens each assign 0.1 probability to Expert B
- Total importance = 9.0 (10 × 0.1 + 90 × 0.1)
```
See that even though the cumulative probability/importance for both experts is the same across a batch, Expert A is very likely to get routed to for those 0.9-probability tokens, while Expert B might not get routed to at all despite having similar importance from a probability perspective! 
So we need to make sure that we add an additional load balancing loss to ensure that the actual token distribution is balanced!

----

To solve this problem, we can create a single auxiliary loss term that captures BOTH:
- Expert importance
- Load Balancing

Defined as the equal routing of tokens between each of the experts. In  [[Switch Transformer]], authors create a loss considering two quantities:
1. The fraction of router ==probabilities== allocated to each expert. (old)
2. The (actual) fraction of tokens ==dispatched== to each expert. (new)
![[Pasted image 20250128145419.png]]
Above: From the [[Switch Transformer]] paper. 
- If we store these two quantities in their own N-dimensional vectors, we can create a single loss term by taking the [[Dot Product]] of these two vectors. 
- ==The resulting loss is minimized when experts receive uniform probability and load balancing==, thus capturing both of our goals within a single auxiliary loss term.

![[Pasted image 20250128153112.png]]
Above: Router Z-Loss: The auxiliary load balancing loss described earlier is widely used throughout the MoE literature, but authors in [[ST-MoE]] propose an extra auxiliary loss term (above), called the router z-loss, which can further improve training stability.
- This router Z-loss constrains the size of the ==logits== (not the probabilities) predicted by the routing mechanism.
- Ideally, we don't want these logits to be too big! But they can be very large, leading to "round-off" errors that can destabilize the training process

-----
Aside: Why is the Z-Loss important for MoE model training stability?

The key issue is above numerical stability when computing softmax probabilities from logits:
1. The routing mechanism produces logits (raw scores) that are passed through a softmax function to get probabilities for each expert.
2. The softmax function involves ==exponentials==: softmax(x) = exp(x) / sum(exp(x))
3. When logits become ==very large, the exponential terms can become extremely large numbers==. This leads to numerical overflow even in float32! This destabilizes training.
4. The router Z-Loss helps prevent these issues by explicitly penalizing large logit values. It encourages the model to produce smaller, more reasonable logit values that:
	1. Won't lead to overflow when exponentiated
	2. Lead to more numerically stable probability computations.
	3. Result in more stable gradient updates during training.
-----

Given that the Z-loss function term shown above focused solely on regularizing the router's logits and performs no load balancing, we typically use the router z-loss in tandem with the auxiliary load balancing loss that we talked about earlier!

Both of these losses are added on top of the LLM's standard language modeling loss:
![[Pasted image 20250128155758.png]]

Nice!


### Expert Capacity

The computation performed in an MoE layer is dynamic due to routing decisions made during both training and inference...
But when we look at most implementations of sparse models, we will see that ==they usually have static batch sizes==.
![[Pasted image 20250128155902.png]]

Expert capacity: To formalize the fixed batch size that we set for each expert, we can define the expert capacity. The expert capacity is defined as shown below.

Expert Capacity = ((Total # tokens in the batch) / N) * Capacity Factor

==The expert capacity defines the maximum number of tokens in a batch that can be sent to each expert.==
- If the number of tokens routed to an expert EXCEEDS the expert capacity, we just ==DROP== those extra tokens!
	- Specifically, we perform no computation for these tokens and let their representation flow directly to the next layer, via the transformer's [[Residual Connection]].

### Capacity Factor

Expert capacity is controlled via the ==capacity factor== setting. 
- A capacity factor of one of one means that tokens are routed in a perfectly-balanced manner across the experts
- Alternatively, setting it *above one* provides extra buffer to accommodate for al imbalance in tokens between experts. However this comes at a cost (higher memory usage and lower efficiency).

How do we set the capacity factor?
- Interestingly, MoE models tend to perform well with relatively low capacity factors.

-------
Aside: Why do we need a capacity factor? 
- 

-------




















