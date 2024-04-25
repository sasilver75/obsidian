---
tags:
  - paper
---
Date: June, 2017 -- Authors include [[Noam Shazeer]], [[Aidan Gomez]], Vaswani
Paper: https://arxiv.org/abs/1706.03762


Includes:
- Scaled dot product attention, with both [[Masked Attention]] and [[Bidirectional Attention]] ([[Self-Attention]] and [[Cross-Attention]] too)
- [[Multi-Headed Attention]] blocks
	- We have *h* different projected versions of queries, keys, and values, and perform the attention function parallel, yielding multiple output values which are then concatenated and once again projected down, resulting in the final values.
- Positional input encoding using sinusoidal embeddings
- [[Encoder-Decoder Architecture]]
- Residual Dropout
- [[Label Smoothing]]

Excerpts
> Attention functions can be described as a mapping of a query and a set of key-value pairs to an output, where the query/key/value/output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Abstract
> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the ==Transformer==, based ==solely on attention mechanisms==, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.


![[Pasted image 20240424225324.png]]

![[Pasted image 20240424225515.png]]

![[Pasted image 20240424230009.png]]
