Types:
- Encoder-only
	- Designed to learn embeddings that can be used for various predictive tasks. Mainly used for learning *representations* of words while taking context into account. 
	- [[Bidirectional Encoder Representations from Transformers]], [[ViT]] models
- Encoder-Decoder
	- These models consist of both an encoder and a decoder; The encoder is responsible for encoding the input sequence into a fixed-length representation, while the decoder generates new texts or answers user queries.
	- Suitable for tasks like translation, summarization, and text generation.
- Decoder-only
	- These models consist of only a decoder, which is trained to predict the next token in a sequence given the previous tokens. Simpler than encoder-decoders. The [[GPT]] series of models are decoder-only, because of their efficiency, scalability, and turing-completeness.
	- Simpler than encoder-decoder models and are already Turing-complete; mainly used for text generations tasks, but can be used for other tasks with prompting.


Variants: 
- [[Vision Transformer]]
- [[Sparse Transformer]]