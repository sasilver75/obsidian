---
aliases:
  - MRL
  - Matryoshka Embedding
---
Date: May 26, 2022 - University of Washington, [[Google Research]], Harvard
Paper: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
(This paper was mildly interesting because it took a while for it to get attention (~18 months))

 Matryoshka dolls, also called "==Russian Nesting Dolls==" are a set of wooden dolls of decreasing size that are placed inside one another. 🪆
- In a similar way, ==Matryoshka embedding models aim to store more important information in earlier dimensions, and less important information in later dimensions.==
	- ==This lets us truncate the original (large) embedding produced by the model, while retaining enough of the information to perform well on downstream tasks!==

Use in Classification
- Matches the accuracy of classifiers based on fixed-feature baselines despite using 14x smaller representation size, on average.

Use in Shortlisting and Reranking
- Rather than performing your downstream task (e.g. ==nearest neighbor search==) on the *full* embeddings, you can shrink ==the embeddings to a smaller size==.
	- ==Afterwards==, you can then ==process the remaining (retrieved) embeddings using their full dimensionality==!

Trade-offs
- Matryoshka models will ==allow you to scale your embedding solutions to your desired storage cost, processing speed, and performance.==


==For Matryoshka Embedding models during training==, a training step *also* involves producing embeddings for your training batch, but then ==you use some loss function to determine *not just the quality of your full-size embeddings*, but *also the quality of your embeddings at various different dimensionalities!*==
- For example, your dimensionalities are 768, 512, 256, 128, 64 -- ==the loss values for each dimensionality are added together, resulting in a final loss value.==
- In practice, this ==incentivizes the model to frontload the most important information at the start of an embedding==, such that it will be retained if the embedding is truncated.
- ==Even at only 8.3% of the embedding size, the Matryoshka model preserves 98.37% of the performance, much higher than the 96.46% by the standard model==. ((On some specific HuggingFace blog post test))


Abstract
> Learned representations are a central component in modern ML systems, serving a multitude of downstream tasks. When training such representations, it is often the case that computational and statistical constraints for each downstream task are unknown. In this context, rigid fixed-capacity representations can be either over or under-accommodating to the task at hand. This leads us to ask: ==can we design a flexible representation that can adapt to multiple downstream tasks with varying computational resources==? Our main contribution is ==Matryoshka Representation Learning (MRL)== which ==encodes information at different granularities and allows a single embedding to adapt to the computational constraints of downstream tasks.== MRL minimally modifies existing representation learning pipelines and imposes no additional cost during inference and deployment. MRL learns coarse-to-fine representations that are at least as accurate and rich as independently trained low-dimensional representations. The flexibility within the learned Matryoshka Representations offer: (a) up to ==14× smaller embedding size for ImageNet-1K classification at the same level of accuracy==; (b) up to ==14× real-world speed-ups for large-scale retrieval== on ImageNet-1K and 4K; and (c) up to ==2% accuracy improvements for long-tail few-shot classification==, all while being as robust as the original representations. Finally, we show that MRL extends seamlessly to web-scale datasets (ImageNet, JFT) across various modalities – vision (ViT, ResNet), vision + language (ALIGN) and language (BERT). MRL code and pretrained models are open-sourced at https://github.com/RAIVNLab/MRL.
> 
![[Pasted image 20240425131244.png]]
![[Pasted image 20240425131229.png]]
![[Pasted image 20240425131219.png]]
![[Pasted image 20240425131210.png]]
![[Pasted image 20240425131052.png]]
FF = "Fixed Feature" representations; We can see that our single MRL vector representation is about 