June 12, 2017 -- [[Google Research]]
Paper: [Attention is all you need](https://arxiv.org/abs/1706.03762)
#zotero 
Takeaway: Dispense with recurrence -- Query, Key, and Value are all you need? (*Also position embeddings, multiple attention heads, feed-forward layers, skip-connections, normalization, etc.)

----



Types:
- Encoder-only
	- Designed to learn embeddings that can be used for various predictive tasks. Mainly used for learning *representations* of words while taking context into account. 
	- [[Bidirectional Encoder Representations from Transformers]], [[Vision Transformer|ViT]] models
- Encoder-Decoder
	- These models consist of both an encoder and a decoder; The encoder is responsible for encoding the input sequence into a fixed-length representation, while the decoder generates new texts or answers user queries.
	- Suitable for tasks like translation, summarization, and text generation.
- Decoder-only
	- These models consist of only a decoder, which is trained to predict the next token in a sequence given the previous tokens. Simpler than encoder-decoders. The [[GPT]] series of models are decoder-only, because of their efficiency, scalability, and turing-completeness.
	- Simpler than encoder-decoder models and are already Turing-complete; mainly used for text generations tasks, but can be used for other tasks with prompting.



Variants: 
- [[Vision Transformer]]
- [[Sparse Transformer]]

----

![[Pasted image 20240130162105.png]]
![[Pasted image 20240130162320.png]]

![[Pasted image 20240130162218.png]]
