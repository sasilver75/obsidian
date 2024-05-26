---
aliases:
  - Retrieval-Enhanced Transformer
---
December 8, 2021 (18 months after GPT-3)
Paper: [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)
[[DeepMind]]
#zotero 
Takeaway: By retrieving over an unprecedented *2T token* database, RETRO obtains comparable performance to GPT-3, despite using 25x fewer parameters (at the time, retrieval for LM work usually considered small transformers and databases of limited size). The generation component is pretrained from scratch while incorporating input from the already-trained, frozen retrieval component.
Combines a frozen BERT retriever, a differentiable encoder, and chunked cross-attention to generate output. Notably, it does retrieval throughout the entire pre-training stage, not just during inference. Furthermore, fetches relevant documents based on chunks of input, allowing for finer-grained, repeated retrieval during generation, instead of only retrieving once per query.

----

Notes
- Uses ==Chunked Cross Attention== to incorporate the retrieved text. The retrieved tokens are fed into an encoder Transformer which outputs a representation. In the decoder, we *interleave* standard Transformer blocks with Retro-blocks that perform cross-attention to the retrieval encoder.
	- We split a given intermediate activation in the decoder into *attending chunks* (see fig 2). We compute the cross-attention between these chunks and... something from the encoded retrieval set ((?)).
	- ==I don't really understand this chunked cross-attention mechanism==
- ==Similar== to [[Fusion-in-Decoder|FiD]], [[RETRO]] processes the retrieved neighbors separately in the encoder, and assemble them in the chunked cross-attention. This ==differs== from (eg) [[REALM]], where they prepend retrieved documents the prompt.
- Retriever is based on a pre-trained, *frozen* [[Bidirectional Encoder Representations from Transformers|BERT]] model. It's interesting that *they ==didn't choose to jointly train the retriever==*, like [[Retrieval-Augmented Generation (Model)|RAG]] did (where they found that it was a useful aspect of the system, I believe?).
	- ==Explanation==: "We use a frozen model to avoid having to periodically recompute embeddings over the entire database during training." Good point!
- ==Retrieval is done during the whole pre-training process in RETRO== -- it's not simply plugged in to solve a certain downstream task.

Abstract
> ==We enhance auto-regressive language models by conditioning on document chunks retrieved from a large corpus==, based on local similarity with preceding tokens. With a 2 trillion token database, our ==Retrieval-Enhanced Transformer (RETRO)== obtains ==comparable performance to GPT-3 and Jurassic-1 on the Pile, despite using 25× fewer parameters==. After fine-tuning, RETRO performance translates to downstream knowledge-intensive tasks such as question answering. RETRO combines a ==frozen Bert retriever==, a ==differentiable encoder== and a ==chunked cross-attention== mechanism to predict tokens based on an order of magnitude more data than what is typically consumed during training. We typically train RETRO from scratch, yet can also rapidly RETROfit pre-trained transformers with retrieval and still achieve good performance. Our work opens up new avenues for improving language models through explicit memory at unprecedented scale.

# Paper Figures

![[Pasted image 20240501170200.png]]
Above: I believe that "non-embedding params" refers to the number of params of the "generator" component in the system. It's interesting by how much performance seems to be improving as the retrieval dataset increases above 500B tokens or so -- I know that it's a log scale, but it seems to indicate that there might still be a large amount of juice? It does seem like increasing the number of retrieved documents has diminishing returns on a log scale.

![[Pasted image 20240501172623.png]]
Above: 
- Note that the input tokens make it both into the encoder (via the query encoder in the kNN retriever) and the decoder as input. For MT tasks with Encoder-decoder architectures, I usually only think of the input tokens as being put into the encoder, rather than the decoder.

![[Pasted image 20240501174232.png]]
Comparison with [[Retrieval-Augmented Generation (Model)|RAG]], [[REALM]], [[DPR]], kNN-LM, and [[Fusion-in-Decoder]] (FiD)

![[Pasted image 20240501174523.png]]

