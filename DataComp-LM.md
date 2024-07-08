---
aliases:
  - DCLM
---
June 17, 2024
59 authors from many universities and research groups, including [[Ludwig Schmidt]]
[DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794v1)
Website link: [DataComp-LM](https://www.datacomp.ai/dclm/index.html#home)

A benchmark/competition introduced in 2024 focused on data-centric AI. Whereas historically we treat datasets as constant and compete on model architectures (eg [[ImageNet|ILSVRC]]), DataComp is a competition in which architecture and training code is held constant, and competitors are encouraged to find ways of filtering and augmenting a dataset of 240T tokens from [[Common Crawl]] (or bringing their own data). The benchmark consists of multiple scales, with various candidate pool sizes and associated compute budgets ranging from 412M to 7B parameters.

---

Abstract
> We introduce ==DataComp for Language Models (DCLM)==, a testbed for controlled dataset experiments with the goal of improving language models. As part of DCLM, we ==provide a standardized corpus of 240T tokens extracted from Common Crawl, effective pretraining recipes based on the OpenLM framework, and a broad suite of 53 downstream evaluations==. ==Participants in the DCLM benchmark can experiment with data curation strategies such as deduplication, filtering, and data mixing at model scales ranging from 412M to 7B parameters==. As a baseline for DCLM, we conduct extensive experiments and find that model-based filtering is key to assembling a high-quality training set. The resulting dataset, DCLM-Baseline enables training a 7B parameter language model from scratch to 64% 5-shot accuracy on MMLU with 2.6T training tokens. Compared to MAP-Neo, the previous state-of-the-art in open-data language models, DCLM-Baseline represents a 6.6 percentage point improvement on MMLU while being trained with 40% less compute. Our baseline model is also comparable to Mistral-7B-v0.3 and Llama 3 8B on MMLU (63% & 66%), and performs similarly on an average of 53 natural language understanding tasks while being trained with 6.6x less compute than Llama 3 8B. ==Our results highlight the importance of dataset design for training language models and offer a starting point for further research on data curation.==