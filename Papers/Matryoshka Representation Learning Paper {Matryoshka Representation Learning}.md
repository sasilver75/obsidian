May 26, 2022 -- UW, [[Google Research]], Harvard
Link [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147?utm_source=substack&utm_medium=email)

This is an interesting paper because it took about a year after it was published for it to get some attention.

Excerpts
> Perhaps due to the inductive bias of gradient-based training, deep learning models tend to diffuse "information" across the entire representation vector... By encoding coarse-to-fine-grained representations which are as accurate as independently-trained counterparts, we can learn with minimal overhead a representation that can be deployed *adaptively* at no additional cost during inference.




Abstract
> ==Learned representations are a central component== in modern ML systems, ==serving a multitude of downstream tasks==. When training such representations, it is often the case that computational and statistical constraints for each downstream task are unknown. In this context ==rigid, fixed capacity representations can be either over or under-accommodating to the task at hand==. This leads us to ask: ==can we design a flexible representation that can adapt to multiple downstream tasks with varying computational resources==? Our main contribution is ==Matryoshka Representation Learning== (MRL) which ==encodes information at different granularities and allows a single embedding to adapt to the computational constraints of downstream tasks==. MRL minimally modifies existing representation learning pipelines and imposes no additional cost during inference and deployment. MRL ==learns coarse-to-fine representations== that are at least as accurate and rich as independently trained low-dimensional representations. The ==flexibility== within the learned Matryoshka Representations offer: (a) up to 14x smaller embedding size for ImageNet-1K classification at the same level of accuracy; (b) up to 14x real-world speed-ups for large-scale retrieval on ImageNet-1K and 4K; and (c) up to 2% accuracy improvements for long-tail few-shot classification, all while being as robust as the original representations. Finally, we show that MRL extends seamlessly to web-scale datasets (ImageNet, JFT) across various modalities -- vision (ViT, ResNet), vision + language (ALIGN) and language (BERT). MRL code and pretrained models are open-sourced at this https URL.


![[Pasted image 20240424233115.png]]

