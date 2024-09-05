---
aliases:
  - DataComp-CLIP
---


April 27, 2023
34 authors from various schools and groups, incl. [[Ludwig Schmidt]]
[DataComp: In search of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108)
Website link: [DataComp-CLIP](https://www.datacomp.ai/dcclip/index.html#home)

A benchmark/competition introduced in 2023 focused on data-centric AI. Whereas historically we treat datasets as constant and compete on model architectures (eg [[ImageNet|ILSVRC]]), DataComp is a competition in which architecture and training code is held constant, and competitors are encouraged to find ways of filtering and augmenting a multimodal 12.8B pair image-text dataset (called "CommonPool") pulled from [[Common Crawl]] (or by bringing their own data)

(Note: This is often called [DataComp-CLIP](https://www.datacomp.ai/dcclip/index.html#home) to better disambiguate between the later-introduced [[DataComp-LM]] competition for language models)

----


Abstract
> Multimodal datasets are a critical component in recent breakthroughs such as Stable Diffusion and GPT-4, yet their design does not receive the same research attention as model architectures or training algorithms. To address this shortcoming in the ML ecosystem, we introduce ==DataComp==, a ==testbed for dataset experiments centered around a new candidate pool of 12.8 billion image-text pairs from Common Crawl==. ==Participants in our benchmark design new filtering techniques or curate new data== sources and then ==evaluate their new dataset by running our standardized CLIP training code and testing the resulting model on 38 downstream test sets==. Our benchmark consists of multiple compute scales spanning four orders of magnitude, which enables the study of scaling trends and makes the benchmark accessible to researchers with varying resources. Our baseline experiments show that the DataComp workflow leads to better training sets. In particular, our best baseline, DataComp-1B, enables training a CLIP ViT-L/14 from scratch to 79.2% zero-shot accuracy on ImageNet, outperforming OpenAI's CLIP ViT-L/14 by 3.7 percentage points while using the same training procedure and compute. We release DataComp and all accompanying code at [this http URL](http://www.datacomp.ai/).
> 


# Paper Figures


# Non-Paper Figures
![[Pasted image 20240707202526.png]]

