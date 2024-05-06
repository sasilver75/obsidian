---
aliases:
  - MoE
---
January 23, 2017 (5 months before *Attention is all you need*)
[[Google Research]] - Including [[Noam Shazeer]], [[Quoc Le]], [[Jeff Dean]], [[Geoff Hinton]]
Paper: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
#zotero 
Takeaway: They introduce an MoE layer to the stacked LSTM context, where an MoE layer consists of a gating network and a collection of experts. The gating network outputs a *sparse* vector, selecting k of the experts to activate during the forward pass for this token. There are some questions about how to balance the gating networks and how to distribute experts across machines.

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