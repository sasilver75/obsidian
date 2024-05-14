November 1, 2019
[[Meta AI Research]]
Paper: [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359)

Created from [[Common Crawl]] using the [[fastText]] data processing tool

Abstract
> Pre-training text representations have led to significant improvements in many areas of natural language processing. The quality of these models benefits greatly from the size of the pretraining corpora as long as its quality is preserved. In this paper, ==we describe an automatic pipeline to extract massive high-quality monolingual datasets from Common Crawl== for a variety of languages. Our pipeline ==follows the data processing introduced in fastText== (Mikolov et al., 2017; Grave et al., 2018), that deduplicates documents and identifies their language. We augment this pipeline with a filtering step to select documents that are close to high quality corpora like Wikipedia.