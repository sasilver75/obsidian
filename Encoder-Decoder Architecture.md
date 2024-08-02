A transformer encoder with a transformer decoder performing [[Cross-Attention]] to the encoder stack.

Examples:
- [[Transformer]]
- [[Transformer-XL]]
- [[T5]]
- [[FLAN-T5]]
- [[BART]]

References:
- [[What happened to BERT and T5? On Transformer Encoders, PrefixLM, and Denoising Objectives (July 16) {Yi Tay}]]

In a sense, it might be possible to consider multimodal models like [[LLaVA]] and [[CogVLM]] as being types of encoder-decoder models, where a vision encoder (eg a [[Vision Transformer|ViT]] or [[Convolutional Neural Network|CNN]]) process image data, which is then cross-attended to by a language model decoder component.