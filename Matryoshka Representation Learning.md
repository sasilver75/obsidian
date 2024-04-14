---
aliases:
  - MRL
  - Matryoshka Embedding
---
Date: May 26, 2022
Paper: https://arxiv.org/abs/2205.13147
(This paper was mildly interesting because it took a while for it to get attention (~18 months))


- Matryoshka dolls, also called "==Russian Nesting Dolls==" are a set of wooden dolls of decreasing size that are placed inside one another.
- In a similar way, ==Matryoshka embedding models aim to store more important information in earlier dimensions, and less important information in later dimensions.==
	- ==This lets us truncate the original (large) embedding produced by the model, while retaining enough of the information to perform well on downstream tasks!==


1. Shortlisting and Reranking
	- Rather than performing your downstream task (e.g. ==nearest neighbor search==) on the *full* embeddings, you can shrink ==the embeddings to a smaller size==.
		- ==Afterwards==, you can then ==process the remaining (retrieved) embeddings using their full dimensionality==!
2. Trade-offs
	- Matryoshka models will ==allow you to scale your embedding solutions to your desired storage cost, processing speed, and performance.==



==For Matryoshka Embedding models during training==, a training step *also* involves producing embeddings for your training batch, but then ==you use some loss function to determine *not just the quality of your full-size embeddings*, but *also the quality of your embeddings at various different dimensionalities!*==
- For example, your dimensionalities are 768, 512, 256, 128, 64 -- ==the loss values for each dimensionality are added together, resulting in a final loss value.==
- In practice, this ==incentivizes the model to frontload the most important information at the start of an embedding==, such that it will be retained if the embedding is truncated.



- ==Even at only 8.3% of the embedding size, the Matryoshka model preserves 98.37% of the performance, much higher than the 96.46% by the standard model==. ((On some specific HuggingFace blog post test))

Paper: https://arxiv.org/abs/2205.13147


