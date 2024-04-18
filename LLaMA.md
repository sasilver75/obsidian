February 27, 2023
Paper: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)("Large Language Model Meta AI")

Significance: [[Meta AI Research]] is the only company to go into [detail](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)  about their experience training a 175B-parameter model.

Abstract
> We introduce LLaMA, a ==collection of foundation language models ranging from 7B to 65B parameters==. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, ==LLaMA-13B outperforms GPT-3 (175B) on most benchmarks==, and ==LLaMA-65B is competitive with the best models, [[Chinchilla]]-70B and [[PaLM]]-540B==. We release all our models to the research community.

 Combined a bunch of the best features from [[PaLM]] and [[Chinchilla]]:
	- Pre-normalize the input of each transformer sub-layer
	- Use [[RMSNorm]] instead of [[LayNorm]]
	- [[SwiGLU]] activation function from [[PaLM]]
	- Uses [[Rotary Positional Embedding|RoPE]], as [[PaLM]] did.
	- Uses [[AdamW]], and [[Chinchilla]] did.

Followed by: [[LLaMA 2]]

![[Pasted image 20240417224708.png]]