April 16, 2023 (5 months after ChatGPT, 2 months after LLaMA) -- [[Together AI]] (and others in collaboration) 
Blog: [RedPajama](https://www.together.ai/blog/redpajama)

"RedPajama, a project to create leading open-source models, starts by reproducing [[LLaMA]] training dataset of over 1.2 trillion tokens. It follows the recipe described in the LLaMA paper."

The full RedPajama dataset has ==1.2T tokens==
It consists of seven data slices:
1. CommonCrawl (5 dumps, published with a [[CCNet]] pippeline)
2. C4 (Standard C4 dataset)
3. GitHub
4. arXiv
5. Books
6. Wikipedia
7. StackExchange

For each data slice, we conduct careful data pre-processing and filtering, tuning our quality filter to roughly match the number of tokens as reported by [[Meta AI Research]] in [[LLaMA]] paper.

Summary
> We are launching RedPajama, an effort to produce a reproducible, fully-open, leading language model. RedPajama is a collaboration between Together, [Ontocord.ai](https://www.ontocord.ai/), [ETH DS3Lab](https://ds3lab.inf.ethz.ch/), [Stanford CRFM](https://crfm.stanford.edu/), and [Hazy Research](https://hazyresearch.stanford.edu/). RedPajama has three key components:
> 1. Pre-training data, which needs to be both high quality and have broad coverage
> 2. Base models, which are trained at scale on this data
> 3. Instruction tuning data and models, which improve the base model to make it usable and safe
> Our starting point is [LLaMA](https://arxiv.org/abs/2302.13971), which is the leading suite of open base models for two reasons: First, LLaMA was trained on a very large (1.2 trillion tokens) dataset that was carefully filtered for quality. Second, the 7 billion parameter LLaMA model is trained for much longer, well beyond the Chincilla-optimal point, to ensure the best quality at that model size. A 7 billion parameter model is particularly valuable for the open community as it can run on a wide variety of GPUs, including many consumer grade GPUs. However, LLaMA and all its derivatives (including Alpaca, Vicuna, and Koala) are only available for non-commercial research purposes. We aim to create a fully open-source reproduction of LLaMA, which would be available for commercial applications, and provide a more transparent pipeline for research.

![[Pasted image 20240419224729.png]]


