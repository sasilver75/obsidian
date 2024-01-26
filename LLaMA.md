Significance: [[Meta AI Research]] is the only company to go into [detail](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)  about their experience training a 175B-parameter model.

 Combined a bunch of the best features from [[PaLM]] and [[Chinchilla]]:
	- Pre-normalize the input of each transformer sub-layer
	- Use [[RMSNorm]] instead of [[LayNorm]]
	- [[SwiGLU]] activation function from [[PaLM]]
	- Uses [[Rotary Positional Embedding|RoPE]], as [[PaLM]] did.
	- Uses [[AdamW]], and [[Chinchilla]] did.

Followed by: [[LLaMA 2]]