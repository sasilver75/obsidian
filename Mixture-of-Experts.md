---
aliases:
  - MoE
---
January 23, 2017 (5 months before *Attention is all you need*)
[[Google Research]] - Including [[Noam Shazeer]], [[Quoc Le]], [[Jeff Dean]], [[Geoff Hinton]]
Paper: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
#zotero 
Takeaway: ...

---

Notes:
- Challenges of conditional computation (in which a subset of the neurons are active, on a forward pass):
	1. Modern computing devices (especially GPUs) are much faster at arithmetic than branching... Most works recognize this and propose turning on/off large chunks of the network with each gating decision.
	2. ==Large batch sizes are critical for performance, but conditional computation reduces the batch sizes for the conditionally active chunks of the network==.
	3. Network bandwidth can be a bottleneck in a multi-GPU setup where we need to send embeddings around the network.
	4. Depending on the scheme, loss terms might be necessary to achieve the desired level of sparsity per-chunk and/or per example. These issues affect model quality and load balancing.
	5. Model capacity is most critical for very 


Examples (roughly chronological):
- Hinton/Jordan's "Adaptive Mixtures of Local Experts"
- [[Noam Shazeer]]'s 2017 "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- [[GShard]]
- [[Switch Transformer]]
- [[GLaM]]

Abstract
> The capacity of a neural network to absorb information is limited by its number of parameters. ==Conditional computation==, where ==parts of the network are active on a per-example basis==, has been proposed in theory ==as a way of dramatically increasing model capacity without a proportional increase in computation==. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, ==achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency== on modern GPU clusters. We introduce a ==Sparsely-Gated Mixture-of-Experts layer (MoE)==, consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.


# Paper Figures
![[Pasted image 20240505111024.png]]
Above: Because this paper came out before the Transformer, this is shown in the context of RNNs. But the MoE block is the same (as far as I can tell) as ones we might use in a Transformer, though the Transformer MoE blocks often will have a FFNN layer after combining results from experts.

# Non-Paper Figures